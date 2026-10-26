import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from w2gg.utils.comm import get_world_size, synchronize, get_rank
from w2gg.utils.metric_logger import MetricLogger
from w2gg.engine.inference import inference
from w2gg.modeling.w2gg import init_glance_memory


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_sampler_epoch(data_loader, epoch):
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)

    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss

    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if get_rank() == 0:
            loss /= world_size

    loss = loss.item()
    return loss


# def is_larger(result1, result2, keys=["Avg_R1"]):
#     if len(keys) == 1:
#         return result1[keys[0]] > result2[keys[0]]
#     elif len(keys) > 1:
#         sum1 = sum(result1[key] for key in keys)
#         sum2 = sum(result2[key] for key in keys)
#         return sum1 > sum2
#     else:
#         raise RuntimeError("keys cannot be null!")


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=5,
):
    logger = logging.getLogger("w2gg.trainer")
    logger.info("Start training")

    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()

    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)

    best_result = {"Avg_dR1": 0.0}
    sample_seed = 0

    # Initialize temporal center memory only for dynamic DGA variants.
    real_model = unwrap_model(model)
    num_clip = cfg.MODEL.W2GG.NUM_CLIPS
    if real_model.uses_dynamic_dga:
        set_sampler_epoch(data_loader, sample_seed)
        memory = init_glance_memory(
            data_loader,
            num_clip,
            dga_mode=real_model.dga_mode,
            continuous_num=real_model.continuous_num,
        )
        real_model.init_memory(memory)
        logger.info(
            "Initialized dynamic DGA memory: mode=%s, continuous_num=%d",
            real_model.dga_mode,
            real_model.continuous_num,
        )
    else:
        logger.info("DGA mode %s does not require temporal memory", real_model.dga_mode)

    model.train()

    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        # 关键修改：
        # 单卡 RandomSampler 没有 set_epoch；
        # 多卡 DistributedSampler 才有 set_epoch。
        set_sampler_epoch(data_loader, sample_seed)

        if epoch <= cfg.SOLVER.FREEZE_BERT:
            for param in param_dict["bert"]:
                param.requires_grad_(False)
        else:
            for param in param_dict["bert"]:
                param.requires_grad_(True)

        logger.info(
            "Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
                str(param_dict["bert"][0].requires_grad),
            )
        )

        logger.info("Training objective: SA-GCL weighted group contrastive loss")

        for iteration, (batches, idx) in enumerate(data_loader):
            iteration += 1

            batches = batches.to(device)
            optimizer.zero_grad()

            loss = model(batches, cur_epoch=epoch, idxs=idx)#forward方法

            meters.update(loss=loss.detach())

            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            meters.update(time=batch_time)

            eta_seconds = meters.time.global_avg * (
                max_iteration - iteration + rest_epoch_iteration
            )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated()
                        / 1024.0
                        / 1024.0,
                    )
                )

            gc.collect()

        scheduler.step()

        # save last ckpt
        checkpointer.save("model_last", **arguments)

        if epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST:
            synchronize()
            torch.cuda.empty_cache()

            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )

            if (
                get_rank() == 0
                and result_dict["Avg_dR1"] > best_result["Avg_dR1"]
            ):
                best_result = dict(result_dict)
                best_result["epoch"] = epoch

                logger.info(
                    "Got a new best result! "
                    f"Avg_dR1={best_result['Avg_dR1'] * 100:.2f}, "
                    f"epoch={epoch}"
                )

                checkpointer.save("best_model", **arguments)

            synchronize()
            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str,
            total_training_time / max_iteration,
        )
    )
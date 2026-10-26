import argparse
import os
import torch
from w2gg.config import cfg
from w2gg.data import make_data_loader
from w2gg.engine.inference import inference
from w2gg.modeling import build_model
from w2gg.utils.checkpoint import CheckpointManager
from w2gg.utils.comm import synchronize, get_rank
from w2gg.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="W2GG")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    #parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger("w2gg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = CheckpointManager(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)[0]
    dataset_name = dataset_names[0] if isinstance(dataset_names, (list, tuple)) else dataset_names
    _ = inference(
        cfg,
        model,
        data_loaders_val,
        dataset_name=dataset_name,
        nms_thresh=cfg.TEST.NMS_THRESH,
        device=cfg.MODEL.DEVICE,
        save_predictions=True,
    )
    synchronize()

if __name__ == "__main__":
    main()

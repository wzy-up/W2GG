from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import os
import torch
import json
from w2gg.data.datasets.utils import iou, score2d_to_moments_scores
from w2gg.utils.comm import is_main_process


def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
    return moments[~suppressed]


def _to_float_pair(moment):
    """Convert a 2-element tensor/list moment to JSON-safe [start, end]."""
    if torch.is_tensor(moment):
        moment = moment.detach().cpu().tolist()
    return [float(moment[0]), float(moment[1])]


def _safe_dataset_name(dataset_name):
    """Create a filesystem-safe short dataset label for prediction files."""
    name = str(dataset_name)
    for ch in ("/", "\\", " ", "(", ")", "'", '"', ","):
        name = name.replace(ch, "_")
    return name.strip("_") or "dataset"


def _save_sentence_predictions(cfg, dataset_name, records):
    """
    Save per-sentence temporal grounding predictions under the current OUTPUT_DIR.

    The file is intentionally written only when evaluate(..., save_predictions=True)
    is requested, so periodic validation during training does not repeatedly dump
    large JSON files. Final testing in train_net.py/test_net.py enables it.
    """
    output_dir = cfg.OUTPUT_DIR
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    filename = "{}_predictions.json".format(_safe_dataset_name(dataset_name))
    output_path = os.path.join(pred_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger = logging.getLogger("w2gg.inference")
    logger.info(
        "Saved %d sentence-level predictions to: %s",
        len(records),
        output_path,
    )
    return output_path


def evaluate(
    cfg,
    dataset,
    predictions,
    nms_thresh,
    recall_metrics=(1,),
    save_predictions=False,
    prediction_dataset_name=None,
):
    """Evaluate W2GG with CAEM-style dR@n, IoU@m.

    When save_predictions=True, also save one record per sentence containing:
      - video_id
      - sentence_index / sentence
      - video duration
      - ground-truth interval
      - Top-1 predicted interval after NMS
      - Top-1 IoU

    The JSON is written to:
        <cfg.OUTPUT_DIR>/predictions/<dataset>_predictions.json
    """
    if not is_main_process():
        return

    # 仿照 CAEM / 10.20-1 表 2：m ∈ {0.1, 0.3, 0.5, 0.7}
    if cfg.DATASETS.NAME in ["charades", "activitynet"]:
        iou_metrics = (0.1, 0.3, 0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)

    dataset_class_name = dataset.__class__.__name__
    logger = logging.getLogger("w2gg.inference")
    logger.info(
        "Performing {} dR evaluation (Size: {}).".format(
            dataset_class_name, len(dataset)
        )
    )

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)

    num_clips = predictions[0]["contrastive"].shape[-1]

    table = [[
        "dR@{},IoU@{:.01f}".format(
            int(r), float(torch.round(m * 100) / 100)
        )
        for r in recall_metrics
        for m in iou_metrics
    ]]

    dr_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    num_instance = 0

    sentence_predictions = []

    for idx, result2d in tqdm(enumerate(predictions)):  # each video
        score2d = result2d["contrastive"]
        duration = float(dataset.get_duration(idx))
        gt_moments = dataset.get_moment(idx)

        # These methods exist in the Charades/ActivityNet dataset classes.
        video_id = dataset.get_vid(idx) if hasattr(dataset, "get_vid") else str(idx)
        sentences = (
            dataset.get_sentence(idx)
            if hasattr(dataset, "get_sentence")
            else [""] * len(gt_moments)
        )

        for sent_idx, (gt_moment, pred_score2d) in enumerate(
            zip(gt_moments, score2d)
        ):
            num_instance += 1

            candidates, scores = score2d_to_moments_scores(
                pred_score2d, num_clips, duration
            )
            moments = nms(candidates, scores, nms_thresh)

            # 保证 gt_moment 和 moments 在同一个 device 上
            if not torch.is_tensor(gt_moment):
                gt_moment = torch.tensor(gt_moment, device=moments.device)
            else:
                gt_moment = gt_moment.to(moments.device)

            # Save exactly the final Top-1 prediction used by evaluation.
            if save_predictions:
                if moments.numel() > 0:
                    pred_top1 = moments[0]
                    top1_iou = float(iou(pred_top1.unsqueeze(0), gt_moment)[0].item())
                    pred_pair = _to_float_pair(pred_top1)
                else:
                    pred_pair = None
                    top1_iou = 0.0

                sentence_predictions.append(
                    {
                        "dataset_index": int(idx),
                        "video_id": str(video_id),
                        "sentence_index": int(sent_idx),
                        "sentence": str(sentences[sent_idx]) if sent_idx < len(sentences) else "",
                        "duration": duration,
                        "gt_moment": _to_float_pair(gt_moment),
                        "pred_moment": pred_pair,
                        "top1_iou": top1_iou,
                    }
                )

            for i, r in enumerate(recall_metrics):
                r = int(r.item())
                top_moments = moments[:r]

                if top_moments.numel() == 0:
                    continue

                mious = iou(top_moments, gt_moment)

                gt_start = gt_moment[0]
                gt_end = gt_moment[1]

                p_start = top_moments[:, 0]
                p_end = top_moments[:, 1]

                tmp_1 = 1 - torch.abs(p_start - gt_start) / duration
                tmp_2 = 1 - torch.abs(p_end - gt_end) / duration
                tmp_dr = tmp_1 * tmp_2

                for j, t in enumerate(iou_metrics):
                    hit = (mious > t)[:r].any()
                    if hit:
                        dr_x_iou[i][j] += tmp_dr.max()

    dr_x_iou /= num_instance

    table.append([
        "{:.02f}".format(dr_x_iou[i][j] * 100)
        for i in range(num_recall_metrics)
        for j in range(num_iou_metrics)
    ])

    table = AsciiTable(table)
    for i in range(num_recall_metrics * num_iou_metrics):
        table.justify_columns[i] = "center"
    logger.info("\n" + table.table)

    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            key = "dR@{},IoU@{:.01f}".format(
                int(recall_metrics[i]),
                float(torch.round(iou_metrics[j] * 100) / 100),
            )
            result_dict[key] = dr_x_iou[i][j]

    result_dict["Avg_dR1"] = dr_x_iou[0].mean().item()

    if save_predictions:
        save_name = prediction_dataset_name or dataset_class_name
        result_dict["predictions_file"] = _save_sentence_predictions(
            cfg,
            save_name,
            sentence_predictions,
        )

    return result_dict

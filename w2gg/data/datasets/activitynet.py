import json
import logging
import os

import torch
from transformers import DistilBertTokenizer

from .utils import (
    bert_embedding,
    build_pseudo_gmask2d,
    get_vid_feat,
    prepare_pseudo_glances,
)


class ActivityNetDataset(torch.utils.data.Dataset):
    """ActivityNet Captions with the old full path and explicit ablations."""

    def __init__(
        self,
        ann_file,
        feat_file,
        num_pre_clips,
        num_clips,
        selected_frames=None,
        selected_fps_file=None,
        pseudo_keyframe_mode="multi",
        pseudo_max_keyframes=0,
        pseudo_refinement="dedup_filter",
        pseudo_filter_threshold_scale=1.0,
        pseudo_weight_mode="confidence",
        gmask_mode="qpm",
        gmask_keep_ratio=0.42,
        gmask_min_keep=24,
        gmask_margin=3,
    ):
        super().__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        is_train = "train" in os.path.basename(ann_file)

        with open(ann_file, "r") as f:
            annos = json.load(f)
        if not selected_frames or not selected_fps_file:
            raise ValueError("ActivityNet requires selected_frames and selected_fps_file")
        with open(selected_frames, "r") as f:
            selected_items = json.load(f)
        with open(selected_fps_file, "r") as f:
            selected_fps = json.load(f)

        self.annos = []
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        logger = logging.getLogger("w2gg.trainer")
        logger.info(
            "Preparing ActivityNet data: keyframe_mode=%s, max_keyframes=%s, "
            "refinement=%s, weight_mode=%s, gmask_mode=%s, selected_frames=%s",
            pseudo_keyframe_mode,
            pseudo_max_keyframes,
            pseudo_refinement,
            pseudo_weight_mode,
            gmask_mode,
            selected_frames,
        )

        # In the original ActivityNet full-model path, the filtering threshold
        # is based on the number of raw keyframes, even when the weight list is
        # shorter and zip() consumes fewer frame/weight pairs.  Keep that exact
        # behavior only for the default full configuration; ablations continue
        # to use the generic preprocessing semantics.
        use_old_full_filter_count = (
            pseudo_keyframe_mode == "multi"
            and int(pseudo_max_keyframes) == 0
            and pseudo_refinement == "dedup_filter"
            and float(pseudo_filter_threshold_scale) == 1.0
            and pseudo_weight_mode == "confidence"
            and gmask_mode == "qpm"
        )

        effective_counts = []
        for sel_item, (vid, anno) in zip(selected_items, annos.items()):
            duration = float(anno["duration"])
            fps = float(selected_fps[vid]) if is_train else 1.0
            values = sel_item.get("value", [])
            weights = sel_item.get("weights", [])

            moments = []
            sentences = []
            gmasks2d = []
            glances = []
            glance_weights = []
            primary_glances = []

            for sent_idx, (timestamp, sentence) in enumerate(
                zip(anno["timestamps"], anno["sentences"])
            ):
                moments.append(
                    torch.tensor(
                        [max(float(timestamp[0]), 0.0), min(float(timestamp[1]), duration)],
                        dtype=torch.float32,
                    )
                )
                sentences.append(sentence)

                if is_train:
                    # Match old ActivityNet indexing: a missing value entry is an
                    # error instead of silently introducing an anchor at clip 0.
                    raw_glance = values[sent_idx]
                    raw_weight = weights[sent_idx] if sent_idx < len(weights) else []

                    filter_threshold_scale = pseudo_filter_threshold_scale
                    if use_old_full_filter_count and len(raw_glance) > 0:
                        # prepare_pseudo_glances divides this scale by the number
                        # of consumed pairs.  Rescaling restores the old
                        # threshold: 1 / len(raw_glance).
                        paired_count = min(len(raw_glance), len(raw_weight))
                        filter_threshold_scale = (
                            float(paired_count) / float(len(raw_glance))
                        )

                    glance_all, w, primary_glance = prepare_pseudo_glances(
                        raw_glance,
                        raw_weight,
                        fps=fps,
                        num_clips=num_clips,
                        duration=duration,
                        keyframe_mode=pseudo_keyframe_mode,
                        max_keyframes=pseudo_max_keyframes,
                        refinement=pseudo_refinement,
                        filter_threshold_scale=filter_threshold_scale,
                        weight_mode=pseudo_weight_mode,
                    )
                    gmask2d = build_pseudo_gmask2d(
                        glance_all,
                        w,
                        num_clips,
                        mode=gmask_mode,
                        keep_ratio=gmask_keep_ratio,
                        min_keep=gmask_min_keep,
                        margin=gmask_margin,
                    )
                    effective_counts.append(len(set(glance_all)))
                else:
                    gmask2d = torch.zeros(num_clips, num_clips, dtype=torch.bool)
                    glance_all = [0]
                    w = torch.tensor([1.0], dtype=torch.float32)
                    primary_glance = 0

                gmasks2d.append(gmask2d)
                glances.append(torch.tensor(glance_all, dtype=torch.long))
                glance_weights.append(w.float())
                primary_glances.append(int(primary_glance))

            moments = torch.stack(moments)
            gmasks2d = torch.stack(gmasks2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)

            assert moments.size(0) == queries.size(0) == word_lens.size(0)
            assert moments.size(0) == gmasks2d.size(0)
            assert moments.size(0) == len(glances) == len(glance_weights)

            self.annos.append(
                {
                    "vid": vid,
                    "moment": moments,
                    "sentence": sentences,
                    "query": queries,
                    "wordlen": word_lens,
                    "duration": duration,
                    "gmasks2d": gmasks2d,
                    "glances": glances,
                    "glance_weights": glance_weights,
                    "primary_glances": torch.tensor(primary_glances, dtype=torch.long),
                }
            )

        num_pairs = sum(len(item["sentence"]) for item in self.annos)
        avg_effective = sum(effective_counts) / max(len(effective_counts), 1)
        logger.info(
            "ActivityNet statistics: videos=%d, video-sentence pairs=%d, "
            "avg_effective_keyframes=%.3f",
            len(self.annos),
            num_pairs,
            avg_effective,
        )

    def __getitem__(self, idx):
        item = self.annos[idx]
        feat = get_vid_feat(
            self.feat_file, item["vid"], self.num_pre_clips, dataset_name="activitynet"
        )
        return (
            feat,
            item["query"],
            item["wordlen"],
            item["gmasks2d"],
            item["moment"],
            len(item["sentence"]),
            item["glances"],
            item["glance_weights"],
            item["primary_glances"],
            idx,
        )

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]["duration"]

    def get_sentence(self, idx):
        return self.annos[idx]["sentence"]

    def get_moment(self, idx):
        return self.annos[idx]["moment"]

    def get_vid(self, idx):
        return self.annos[idx]["vid"]
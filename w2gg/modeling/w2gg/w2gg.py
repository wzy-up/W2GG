import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    gen_gaussian_weight,
    gen_multi_glances_gaussian_weight,
    gen_static_multi_glances_gaussian_weight,
)
from .feat2d import build_feat2d
from .featpool import build_featpool
from .loss import GroupContrastiveLoss, build_bce_with_weight_loss
from .proposal_conv import build_proposal_conv
from .text_encoder import build_text_encoder


class W2GG(nn.Module):
    DYNAMIC_DGA_MODES = {"single_anchor", "multi_mean", "multi_weighted"}
    VALID_DGA_MODES = DYNAMIC_DGA_MODES | {"single_static", "static_multi"}

    def __init__(self, cfg):
        super().__init__()
        self.joint_space_size = cfg.MODEL.W2GG.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.W2GG.TEXT_ENCODER.NAME
        self.sigma = cfg.SOLVER.SIGMA
        self.thresh = cfg.SOLVER.THRESH
        self.window = cfg.SOLVER.WINDOW
        self.first_time = True

        configured_mode = str(cfg.SOLVER.DGA.MODE)
        if configured_mode == "auto":
            configured_mode = "multi_weighted" if cfg.SOLVER.USE_DGA else "single_static"
        if configured_mode not in self.VALID_DGA_MODES:
            raise ValueError(
                f"Unknown SOLVER.DGA.MODE={configured_mode!r}; "
                f"valid modes are {sorted(self.VALID_DGA_MODES)}"
            )
        self.dga_mode = configured_mode
        self.continuous_num = int(cfg.SOLVER.DGA.CONTINUOUS_NUM)
        self.ema_alpha = float(cfg.SOLVER.DGA.EMA_ALPHA)
        self.fallback_to_top_anchor = bool(cfg.SOLVER.DGA.FALLBACK_TO_TOP_ANCHOR)
        self.center_pos_thresh = float(cfg.SOLVER.CENTER_POS_THRESH)
        self.center_neg_thresh = float(cfg.SOLVER.CENTER_NEG_THRESH)

        self.memory = {}
        self.sim_memory = {}

        self.featpool = build_featpool(cfg)
        self.feat2d = build_feat2d(cfg)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.group_contrastive_loss = GroupContrastiveLoss(cfg, self.feat2d.mask2d)
        self.bce_loss = build_bce_with_weight_loss(cfg, self.feat2d.mask2d)

    @property
    def uses_dynamic_dga(self):
        return self.dga_mode in self.DYNAMIC_DGA_MODES

    def init_memory(self, memory):
        self.memory = memory
        self.sim_memory = {}

    @staticmethod
    def _normalized_anchor_weights(indices, weights, mode, device, dtype):
        """Select anchors while retaining old weighted-DGA normalization."""
        if weights.numel() != indices.numel():
            raw = torch.ones(indices.numel(), device=device, dtype=dtype)
        else:
            raw = weights.to(device=device, dtype=dtype).flatten()
            raw = torch.nan_to_num(
                raw, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(min=0.0)
            if raw.sum() <= 1e-6:
                raw = torch.ones(indices.numel(), device=device, dtype=dtype)

        if mode == "single_anchor":
            top = int(torch.argmax(raw).item())
            return indices[top : top + 1], torch.ones(1, device=device, dtype=dtype)
        if mode == "multi_mean":
            return (
                indices,
                torch.ones(indices.numel(), device=device, dtype=dtype)
                / indices.numel(),
            )
        if mode == "multi_weighted":
            # Exact old expression.  The 1e-6 is intentionally added to the
            # denominator rather than applied through clamp_min.
            return indices, raw / (raw.sum() + 1e-6)
        raise ValueError(f"Unsupported dynamic mode={mode!r}")

    @torch.no_grad()
    def update_memory(self, batches, idxs, feats):
        if not self.uses_dynamic_dga:
            raise RuntimeError(f"update_memory called for non-dynamic mode={self.dga_mode}")
        if not self.memory:
            raise RuntimeError("Dynamic DGA memory is empty. Call init_memory before training.")

        feats_norm = F.normalize(feats, dim=1)
        sims_per_video = []
        num_sent_list = []
        top_positions = []

        for vid_feats, glances_per_video, weights_per_video, primary_glances, idx in zip(
            feats_norm,
            batches.glances,
            batches.glance_weights,
            batches.primary_glances,
            idxs,
        ):
            num_sent = len(glances_per_video)
            sim = torch.full(
                (num_sent, vid_feats.size(1)),
                -1.0,
                device=vid_feats.device,
                dtype=vid_feats.dtype,
            )

            for sent_idx, (indices, weights) in enumerate(
                zip(glances_per_video, weights_per_video)
            ):
                indices = indices.long().flatten().clamp(0, vid_feats.size(1) - 1)
                if indices.numel() == 0:
                    top_positions.append(None)
                    continue

                if self.dga_mode == "single_anchor":
                    primary = int(primary_glances[sent_idx].item())
                    primary = max(0, min(vid_feats.size(1) - 1, primary))
                    active_indices = torch.tensor(
                        [primary], device=vid_feats.device, dtype=torch.long
                    )
                    aggregation_weights = torch.ones(
                        1, device=vid_feats.device, dtype=vid_feats.dtype
                    )
                    top_positions.append(primary)
                else:
                    raw = weights.to(
                        device=vid_feats.device, dtype=vid_feats.dtype
                    ).flatten()
                    if raw.numel() != indices.numel() or raw.sum() <= 1e-6:
                        raw = torch.ones(
                            indices.numel(),
                            device=vid_feats.device,
                            dtype=vid_feats.dtype,
                        )
                    top_positions.append(
                        int(indices[int(torch.argmax(raw).item())].item())
                    )
                    active_indices, aggregation_weights = self._normalized_anchor_weights(
                        indices,
                        weights,
                        self.dga_mode,
                        device=vid_feats.device,
                        dtype=vid_feats.dtype,
                    )

                anchor_feats = vid_feats.index_select(1, active_indices)
                anchor_to_clip = torch.mm(anchor_feats.t(), vid_feats)
                sim[sent_idx] = (
                    aggregation_weights.unsqueeze(1) * anchor_to_clip
                ).sum(dim=0).clamp(min=-1.0, max=1.0)

            # Restore the old first-epoch EMA condition.  The mode switches still
            # select which anchors produced the current similarity.
            if not self.first_time and idx in self.sim_memory:
                previous = self.sim_memory[idx].to(device=sim.device, dtype=sim.dtype)
                if previous.shape == sim.shape:
                    sim = self.ema_alpha * sim + (1.0 - self.ema_alpha) * previous
            self.sim_memory[idx] = sim.detach().cpu()
            sims_per_video.append(sim)
            num_sent_list.append(num_sent)

        sims_all = torch.cat(sims_per_video, dim=0)
        counters_all = torch.cat([self.memory[idx] for idx in idxs], dim=0).to(
            sims_all.device
        )

        counters_all[sims_all >= self.center_pos_thresh] += 1
        negative = (sims_all < self.center_neg_thresh) & (counters_all >= 1)
        counters_all[negative] -= 1

        no_center = (counters_all >= self.continuous_num).sum(dim=1) == 0
        if no_center.any():
            for row in torch.where(no_center)[0].tolist():
                position = None
                if self.fallback_to_top_anchor and row < len(top_positions):
                    position = top_positions[row]
                if position is None:
                    position = int(torch.argmax(sims_all[row]).item())
                counters_all[row, position] = self.continuous_num

        counters_split = torch.split(counters_all, num_sent_list, dim=0)
        sims_split = torch.split(sims_all, num_sent_list, dim=0)
        for batch_idx, idx in enumerate(idxs):
            self.memory[idx] = counters_split[batch_idx].detach().cpu()
        return counters_split, sims_split

    @staticmethod
    def _primary_anchor_tensors(primary_glances):
        return [item.long() for item in primary_glances]

    def _build_gaussian_prior(self, batches, feats, idxs, length):
        if self.dga_mode == "single_static":
            anchors = self._primary_anchor_tensors(batches.primary_glances)
            return gen_gaussian_weight(anchors, length, self.sigma)

        if self.dga_mode == "static_multi":
            return gen_static_multi_glances_gaussian_weight(
                batches.glances,
                batches.glance_weights,
                length,
                self.sigma,
                aggregation="weighted",
            )

        counters, sims = self.update_memory(batches, idxs, feats)
        return gen_multi_glances_gaussian_weight(
            counters,
            sims,
            length,
            self.sigma,
            cnum=self.continuous_num,
        )

    def forward(self, batches, cur_epoch=1, idxs=None):
        if cur_epoch > 1:
            self.first_time = False

        gmasks2d = batches.gmasks2d
        assert len(gmasks2d) == batches.feats.size(0)

        feats = self.featpool(batches.feats)
        map2d = self.feat2d(feats)
        map2d = self.proposal_conv(map2d)
        sent_feat = self.text_encoder(batches.queries, batches.wordlens)

        if self.training:
            if idxs is None:
                raise ValueError("Training forward requires dataset indices for DGA memory")
            length = gmasks2d[0].size(-1)
            weight2d = self._build_gaussian_prior(batches, feats, idxs, length)
            return self.group_contrastive_loss(
                map2d, sent_feat, gmasks2d, weight2d, cur_epoch
            )

        contrastive_scores = []
        for batch_idx, sentence_features in enumerate(sent_feat):
            video_features = F.normalize(map2d[batch_idx], dim=0)
            sentence_features = F.normalize(sentence_features, dim=1)
            _, temporal_size, _ = video_features.size()
            score = torch.mm(
                sentence_features,
                video_features.reshape(video_features.size(0), -1),
            ).reshape(-1, temporal_size, temporal_size)
            contrastive_scores.append(score * self.feat2d.mask2d)
        return contrastive_scores

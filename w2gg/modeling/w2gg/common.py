import math

import torch


def iou(candidates, gts):
    start, end = candidates[:, 0], candidates[:, 1]
    gt_start, gt_end = gts[:, 0], gts[:, 1]
    inter = torch.min(end, gt_end) - torch.max(start, gt_start)
    union = torch.max(end, gt_end) - torch.min(start, gt_start)
    return inter.clamp(min=0).float() / union


def gen_candidate_moments(num_clips):
    map2d = torch.ones(num_clips, num_clips)
    start, end = torch.where(map2d)
    end += 1
    candidates = torch.stack([start, end], dim=1)
    return candidates.view(-1, num_clips, 2)


def _normalize_weights(weights, count, device=None, dtype=torch.float32):
    if torch.is_tensor(weights):
        w = weights.detach().to(device=device, dtype=dtype).flatten()
    else:
        w = torch.tensor(list(weights), device=device, dtype=dtype).flatten()
    if w.numel() != count:
        w = torch.ones(count, device=device, dtype=dtype)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if w.sum() <= 1e-6:
        w = torch.ones(count, device=device, dtype=dtype)
    return w / w.sum().clamp_min(1e-6)


def init_glance_memory(dataloader, length, dga_mode="multi_weighted", continuous_num=1):
    """Initialize temporal center counters for dynamic DGA ablations.

    ``single_anchor`` initializes only the highest-confidence pseudo glance.
    ``multi_mean`` initializes every refined pseudo glance equally.
    ``multi_weighted`` additionally gives above-average confidence anchors one
    extra counter level. Other clips start from zero and can be mined later.
    """
    valid_modes = {"single_anchor", "multi_mean", "multi_weighted"}
    if dga_mode not in valid_modes:
        raise ValueError(f"Memory initialization requires a dynamic DGA mode, got {dga_mode!r}")

    memory = {}
    for batch, idxs in dataloader:
        for glances, sent_weights, primary_glances, idx in zip(
            batch.glances, batch.glance_weights, batch.primary_glances, idxs
        ):
            counter = torch.zeros(len(glances), length, dtype=torch.long)

            for sent_idx, (keyframes, weights) in enumerate(zip(glances, sent_weights)):
                keyframes = keyframes.detach().cpu().long().flatten()
                if keyframes.numel() == 0:
                    raise RuntimeError(
                        f"[W2GG][InitMemory] empty keyframes at sample={idx}, sentence={sent_idx}"
                    )
                if ((keyframes < 0) | (keyframes >= length)).any():
                    raise RuntimeError(
                        f"[W2GG][InitMemory] keyframe out of range at sample={idx}, "
                        f"sentence={sent_idx}, keyframes={keyframes.tolist()}, length={length}"
                    )

                w = _normalize_weights(weights, keyframes.numel())
                if dga_mode == "single_anchor":
                    primary = int(primary_glances[sent_idx].item())
                    selected = [(primary, int(continuous_num) + 1)]
                elif dga_mode == "multi_mean":
                    selected = [
                        (int(k.item()), int(continuous_num)) for k in keyframes
                    ]
                else:
                        mean_w = float(w.mean().item())

                        selected = []
                        for k, wk in zip(keyframes, w):
                            level = 3 if float(wk.item()) >= mean_w else 2
                            selected.append((int(k.item()), level))

                for keyframe, level in selected:
                    counter[sent_idx, keyframe] = max(
                        int(counter[sent_idx, keyframe].item()), level
                    )

            memory[idx] = counter
    return memory


def _gaussian_bank(length, sigma, device, dtype=torch.float32):
    centers = (torch.arange(length, device=device, dtype=dtype) / length) * 2 - 1
    x = torch.linspace(-1, 1, steps=length, device=device, dtype=dtype)
    bank = torch.exp(-((x.unsqueeze(0) - centers.unsqueeze(1)) ** 2) / (2 * sigma**2))
    bank = bank / bank.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    return bank


def _one_dimensional_to_triplet_2d(weight1d):
    
    num_sent, length = weight1d.shape
    begin, end = torch.where(torch.ones(length, length, device=weight1d.device, dtype=torch.bool))
    end_exclusive = end + 1
    mid = ((begin + end_exclusive) / 2).long()
    end_clamped = end_exclusive.clamp(min=0, max=length - 1)
    final = (
        weight1d[:, begin] + weight1d[:, mid] + weight1d[:, end_clamped]
    ) / 3.0
    return final.view(num_sent, length, length)


def gen_gaussian_weight(anchors, length, sigma):
    
    num_sent_list = [item.size(0) for item in anchors]
    anchors_all = torch.cat(anchors, dim=0).long()
    bank = _gaussian_bank(length, sigma, anchors_all.device)
    weight1d = bank.index_select(0, anchors_all.clamp(0, length - 1))
    weight2d = _one_dimensional_to_triplet_2d(weight1d)
    return torch.split(weight2d, num_sent_list, dim=0)


def gen_static_multi_glances_gaussian_weight(
    glances,
    glance_weights,
    length,
    sigma,
    aggregation="weighted",
):
    """Generate a fixed multi-Gaussian mixture without dynamic center mining."""
    if aggregation not in {"weighted", "mean"}:
        raise ValueError(f"Unknown static aggregation={aggregation!r}")

    output = []
    for glances_per_video, weights_per_video in zip(glances, glance_weights):
        sent_weights_1d = []
        device = glances_per_video[0].device if len(glances_per_video) else torch.device("cpu")
        bank = _gaussian_bank(length, sigma, device)

        for keyframes, weights in zip(glances_per_video, weights_per_video):
            keyframes = keyframes.long().flatten().clamp(0, length - 1)
            if keyframes.numel() == 0:
                keyframes = torch.zeros(1, dtype=torch.long, device=device)
            if aggregation == "mean":
                w = torch.ones(keyframes.numel(), device=device) / keyframes.numel()
            else:
                w = _normalize_weights(weights, keyframes.numel(), device=device)
            mixture = (w.unsqueeze(1) * bank.index_select(0, keyframes)).sum(dim=0)
            sent_weights_1d.append(mixture)

        weight1d = torch.stack(sent_weights_1d, dim=0)
        output.append(_one_dimensional_to_triplet_2d(weight1d))
    return output


def gen_multi_glances_gaussian_weight(glance_counters, sims, length, sigma, cnum=1):
    num_sent_list = [item.size(0) for item in glance_counters]
    counters_all = torch.cat(glance_counters, dim=0)
    sims_all = torch.cat(sims, dim=0)
    num_sent_sum = counters_all.size(0)
    device = counters_all[0].device

    centers = (torch.arange(0, length) / length) * 2 - 1
    centers = centers.view(-1, 1).to(device)
    positions = (
        torch.linspace(-1, 1, steps=length)
        .view(1, length)
        .expand(centers.size(0), length)
        .to(device)
    )

    gaussian = torch.exp(-((positions - centers) ** 2) / (2 * sigma**2)) / (
        math.sqrt(2 * math.pi) * sigma
    )
    gaussian /= torch.max(gaussian, dim=1, keepdim=True)[0]

    gaussian_all = gaussian.unsqueeze(dim=0).repeat(num_sent_sum, 1, 1)
    mask = counters_all >= int(cnum)
    mask_num = mask.sum(dim=1, keepdim=True)
    if (mask_num == 0).any():
        bad = int((mask_num == 0).sum().item())
        total = mask_num.numel()
        raise RuntimeError(
            f"[W2GG-FATAL][GaussianWeight] mask_num==0 count={bad}/{total}, stop training."
        )

    gaussian_all = gaussian_all * sims_all.unsqueeze(dim=-1)
    weight1d = gaussian_all * mask.unsqueeze(dim=-1)
    weight1d = weight1d.sum(dim=1) / mask_num

    # Original triplet proposal conversion: begin, midpoint and end.
    begin, end = torch.where(torch.ones(length, length))
    end = end + 1
    mid = ((begin + end) / 2).long()
    begin_weight = weight1d[:, begin]
    mid_weight = weight1d[:, mid]
    end = torch.clip(end, min=0, max=length - 1)
    end_weight = weight1d[:, end]
    weight2d = ((begin_weight + mid_weight + end_weight) / 3.0).view(
        num_sent_sum, length, length
    )
    return torch.split(weight2d, num_sent_list, dim=0)


import os
from os.path import join, exists
import h5py
import numpy as np
import math 
import torch
# import torchtext
from torch.functional import F


def clip_indices_to_gmask2d(glance_indices, num_clips, device=None):
    """
    Build union gmask2d from multiple clip-level pseudo keyframes.

    gmask2d[i, j] = True if moment [i, j] contains at least one pseudo keyframe.
    """
    gmask2d = torch.zeros(num_clips, num_clips, dtype=torch.bool, device=device)

    if torch.is_tensor(glance_indices):
        indices = glance_indices.detach().cpu().long().flatten().tolist()
    else:
        indices = list(glance_indices)

    indices = sorted(set(int(x) for x in indices if 0 <= int(x) < num_clips))

    for g in indices:
        gmask2d[:g + 1, g:] = True

    return gmask2d

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def box_iou(boxes1, boxes2):
    area1 = box_length(boxes1)
    area2 = box_length(boxes2)
    max_start = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    min_end = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M]
    inter = (min_end - max_start).clamp(min=0)  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def box_length(boxes):
    return boxes[:, 1] - boxes[:, 0]


def gen_candidate_moments(num_clips):
    map2d = torch.ones(num_clips, num_clips)
    start, end = torch.where(map2d)   
    end += 1
    candidates = torch.stack([start, end], dim=1)
    return candidates.view(-1, num_clips, 2) # [num_clips, num_clips, 2]
    
    
def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()   
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d
#多帧
# def glance_to_gmask2d(glance, num_clips, duration):
#     glance_all = []
#     gmask2d = torch.zeros(num_clips, num_clips, dtype=torch.bool)
#     glance_ind = round((glance[0] / duration) * (num_clips-1))
#     for i in range(len(glance)):
#         glance_all.append(round((glance[i] / duration) * (num_clips-1)))
#     # mark the moments cover the glance
#     for i in range(0, glance_ind+1):
#         idx = torch.arange(glance_ind, num_clips)
#         gmask2d[i, idx] = 1
#     return gmask2d, glance_all
#单帧
def glance_to_gmask2d(glance, num_clips, duration):
    glance_all = []

    for i in range(len(glance)):
        if glance[i] < 0 or glance[i] > duration:
            raise ValueError(f"glance[{i}] is out of range: {glance[i]}")

        clip_idx = round((glance[i] / duration) * (num_clips - 1))
        clip_idx = max(0, min(num_clips - 1, clip_idx))
        glance_all.append(clip_idx)

    glance_all = sorted(set(glance_all))
    gmask2d = clip_indices_to_gmask2d(glance_all, num_clips)

    return gmask2d, glance_all

def gen_gaussian_weight(anchors, length, sigma_factor):
    """
    Args:
        anchors (list): list[item], item:[num_sent,]
        length (int): video lenght to be generated guassian weight with sigma  
        sigma (float): used for guassian distribution 
    Returns:
        weight_split(list): list[item], item: [num_sent, length, length]
    """
    num_sent_list = [item.size(0) for item in anchors]
    anchors = torch.cat(anchors, dim=0)
    N = anchors.size(0)
    
    # [N, lenght]
    x = torch.linspace(-1, 1, steps=length).view(1, length).expand(N, length)
    sigma = 1.0 * sigma_factor
    
    # normalize glance ind into range [-1, 1]
    u = (anchors / length) * 2 - 1 # [0, 1] -> [0, 2] - 1 -> [-1, 1]
    u = u.view(N, 1)
    
    # compute the weight 
    # [N, length]
    weight = torch.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    weight /= torch.max(weight, dim=1, keepdim=True)[0] # normalize weight 
    
    # generate 2d-map weight 
    weight2d = torch.ones(length, length)
    begin, end = torch.where(weight2d)
    end = end + 1 
    # [length * length]
    mid = ((begin + end) / 2).long()
    mid_weight = weight[:, mid]
    weight2d = mid_weight.view(N, length, -1)
    weight2d = weight2d.to(anchors.device)
    weight2d_split = torch.split(weight2d, num_sent_list, dim=0)
    return weight2d_split 
    

def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)

def maxfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    maxfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            maxfeats.append(feats[s:e].max(dim=0)[0])
        else:
            maxfeats.append(feats[s])
    return torch.stack(maxfeats)
    
def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid][:]
            else:
                feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            vid_feats[vid] = avgfeats(feat, num_pre_clips) 
    return vid_feats

def get_vid_feat(feat_file, vid, num_pre_clips, dataset_name):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "activitynet":
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        elif dataset_name == "charades":
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        elif dataset_name == "tacos":
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        else:
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)

    return avgfeats(feat, num_pre_clips)

def get_feat_didemo(feat_file, vid):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        feat = f[vid][:]
    return torch.from_numpy(feat)

def get_c3d_charades(feat_file, num_pre_clips):
    assert exists(feat_file)
    feat = torch.load(feat_file)
    #feat = F.normalize(feat, dim=1)
    return maxfeats(feat, num_pre_clips)

def bert_embedding(sentence, tokenizer):
    query_token = tokenizer(sentence, return_tensors="pt", padding=True)
    word_lens = query_token['attention_mask'].sum(dim=1)
    queries = query_token['input_ids']
    return queries, word_lens


def glove_embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)

# [修改 2] 新增：把原始 frame index 映射到 clip index。
# 注意：这里保留重复 clip，不在这里去重，因为后面需要把落到同一 clip 的权重相加。
def frames_to_clip_indices(raw_glance, fps, num_clips, duration):
    glance_all = []

    for g in raw_glance:
        if fps <= 0:
            raise ValueError(f"fps must be positive, but got {fps}")

        glance_sec = float(g) / float(fps)

        if glance_sec < 0 or glance_sec > duration:
            raise ValueError(f"glance second is out of range: {glance_sec}, duration={duration}")

        clip_idx = round((glance_sec / duration) * (num_clips - 1))
        clip_idx = max(0, min(num_clips - 1, int(clip_idx)))
        glance_all.append(clip_idx)

    return glance_all


# [修改 3] 新增：用最终有效伪关键帧重新生成 union 版 gmask2d。
# 含义：候选 moment [i, j] 只要包含任意一个有效伪关键帧 g，就置为 True。
# def clips_to_union_gmask2d(glance_indices, num_clips):
#     gmask2d = torch.zeros(num_clips, num_clips, dtype=torch.bool)

#     if torch.is_tensor(glance_indices):
#         indices = glance_indices.detach().cpu().long().flatten().tolist()
#     else:
#         indices = list(glance_indices)

#     indices = sorted(set(int(g) for g in indices if 0 <= int(g) < num_clips))

#     for g in indices:
#         gmask2d[:g + 1, g:] = True

#     return gmask2d


# [修改 4] 新增：安全归一化，避免权重和为 0 或出现空列表时出问题。
def safe_normalize_weights(weights):
    if torch.is_tensor(weights):
        weights = weights.detach().clone().to(dtype=torch.float32, device="cpu")
    else:
        weights = torch.tensor(weights, dtype=torch.float32)

    if weights.numel() == 0:
        return weights

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)

    if weights.sum() <= 1e-6:
        weights = torch.ones_like(weights) / weights.numel()
    else:
        weights = weights / weights.sum()

    return weights


def prepare_pseudo_glances(
    raw_glance,
    raw_weights,
    fps,
    num_clips,
    duration,
    keyframe_mode="multi",
    max_keyframes=0,
    refinement="dedup_filter",
    filter_threshold_scale=1.0,
    weight_mode="confidence",
):
    """Prepare pseudo keyframes while preserving the old full-model path.

    The full configuration (multi + dedup_filter + confidence + no top-K
    truncation) uses the exact W2GG-old operation order:

      frame -> clip -> merge duplicate clips -> normalize -> threshold ->
      divide by ``sum + 1e-6``.

    Other combinations are the explicit ablation branches.  This keeps the
    original full model as the default implementation without removing any
    single/multi, refinement, weighting, or selector-output experiment.
    """
    valid_keyframe_modes = {"single", "multi"}
    valid_refinements = {"none", "dedup", "dedup_filter"}
    valid_weight_modes = {"confidence", "uniform"}
    if keyframe_mode not in valid_keyframe_modes:
        raise ValueError(f"Unknown keyframe_mode={keyframe_mode!r}")
    if refinement not in valid_refinements:
        raise ValueError(f"Unknown refinement={refinement!r}")
    if weight_mode not in valid_weight_modes:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")

    frames = list(raw_glance or [])
    scores = list(raw_weights or [])
    if not frames:
        raise ValueError("At least one pseudo keyframe is required")

    # The original selector output is ordered, but the single-anchor and
    # optional top-K ablations must explicitly choose by confidence.
    if len(scores) != len(frames):
        if (
            keyframe_mode == "multi"
            and int(max_keyframes) == 0
            and refinement == "dedup_filter"
            and weight_mode == "confidence"
        ):
            # Match the old zip semantics exactly in the full path.
            scores = list(scores)
        else:
            scores = [1.0] * len(frames)

    cleaned = []
    for frame, score in zip(frames, scores):
        try:
            frame = int(frame)
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            score = 0.0
        cleaned.append((frame, max(score, 0.0)))
    if not cleaned:
        raise ValueError("No valid pseudo keyframe remains after input validation")

    top_item = max(cleaned, key=lambda item: item[1])
    primary_clip = frames_to_clip_indices(
        [top_item[0]], fps, num_clips, duration
    )[0]

    selected = list(cleaned)
    if int(max_keyframes) > 0:
        selected = sorted(selected, key=lambda item: item[1], reverse=True)[
            : int(max_keyframes)
        ]
    if keyframe_mode == "single":
        selected = [max(selected, key=lambda item: item[1])]

    raw_count = len(selected)
    raw_frames = [item[0] for item in selected]
    raw_scores = [item[1] for item in selected]
    clip_indices = frames_to_clip_indices(raw_frames, fps, num_clips, duration)

    if refinement == "none":
        final_indices = [int(x) for x in clip_indices]
        final_weights = safe_normalize_weights(raw_scores)
    else:
        # Old full-model operation order: merge raw (not pre-normalized) scores.
        clip2weight = {}
        for clip_idx, score in zip(clip_indices, raw_scores):
            clip_idx = int(clip_idx)
            clip2weight[clip_idx] = clip2weight.get(clip_idx, 0.0) + float(score)

        final_indices = sorted(clip2weight.keys())
        final_weights = safe_normalize_weights(
            [clip2weight[clip_idx] for clip_idx in final_indices]
        )

        if refinement == "dedup_filter":
            threshold = float(filter_threshold_scale) / float(raw_count)
            keep = final_weights >= threshold
            if not bool(keep.any()):
                keep[torch.argmax(final_weights)] = True
            index_tensor = torch.tensor(final_indices, dtype=torch.long)
            final_indices = index_tensor[keep].tolist()
            final_weights = final_weights[keep]

    if not final_indices or final_weights.numel() == 0:
        raise ValueError("No valid pseudo keyframe remains after refinement")

    if weight_mode == "uniform":
        final_weights = torch.ones(len(final_indices), dtype=torch.float32)
        final_weights = final_weights / final_weights.sum()
    else:
        # Preserve the old final numerical normalization for the full path.
        final_weights = final_weights / (final_weights.sum() + 1e-6)

    return final_indices, final_weights.float(), int(primary_clip)


def _proposal_quality_score(
    glance_indices,
    glance_weights,
    num_clips,
    mode,
    margin,
):
    """Return a dense proposal quality score and candidate mask."""
    indices = [int(g) for g in glance_indices if 0 <= int(g) < num_clips]
    if not indices:
        empty = torch.zeros(num_clips, num_clips, dtype=torch.float32)
        return empty, empty.bool()

    weights = safe_normalize_weights(glance_weights)
    if weights.numel() != len(indices):
        weights = torch.ones(len(indices), dtype=torch.float32) / len(indices)

    start = torch.arange(num_clips).view(num_clips, 1).expand(num_clips, num_clips)
    end = torch.arange(num_clips).view(1, num_clips).expand(num_clips, num_clips)
    valid = start <= end

    union = torch.zeros(num_clips, num_clips, dtype=torch.float32)
    coverage = torch.zeros_like(union)
    for g, w in zip(indices, weights):
        inside = (start <= g) & (end >= g) & valid
        union[inside] = 1.0
        coverage[inside] += float(w)

    length = (end - start + 1).float().clamp(min=1.0)
    pseudo_span = max(indices) - min(indices) + 1
    allow_len = min(num_clips, pseudo_span + int(margin))
    length_penalty = torch.clamp(float(allow_len) / length, max=1.0)

    if mode == "confidence":
        score = coverage
    elif mode == "length":
        score = length_penalty.clone()
    elif mode == "qpm":
        score = coverage * length_penalty
    else:
        raise ValueError(f"Quality score is not defined for mode={mode!r}")

    score[~valid] = 0.0
    candidate_mask = valid if mode == "length" else union.bool() & valid
    return score, candidate_mask


def build_pseudo_gmask2d(
    glance_indices,
    glance_weights,
    num_clips,
    mode="qpm",
    keep_ratio=0.5,
    min_keep=24,
    margin=3,
):
    """Build proposal supervision masks for full and ablation configurations.

    ``qpm`` is the original W2GG implementation.  ``single``, ``union``,
    ``confidence`` and ``length`` remain explicit ablation variants.
    """
    valid_modes = {"single", "union", "confidence", "length", "qpm"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown pseudo gmask mode={mode!r}")

    indices = [int(g) for g in glance_indices if 0 <= int(g) < num_clips]
    if not indices:
        return torch.zeros(num_clips, num_clips, dtype=torch.bool)

    weights = safe_normalize_weights(glance_weights)
    if weights.numel() != len(indices):
        weights = torch.ones(len(indices), dtype=torch.float32) / len(indices)

    if mode == "single":
        top = int(torch.argmax(weights).item())
        return clip_indices_to_gmask2d([indices[top]], num_clips)
    if mode == "union":
        return clip_indices_to_gmask2d(indices, num_clips)

    score, candidate_mask = _proposal_quality_score(
        indices, weights, num_clips, mode=mode, margin=margin
    )
    candidate_num = int(candidate_mask.sum().item())
    if candidate_num == 0:
        mask = torch.zeros(num_clips, num_clips, dtype=torch.bool)
        for g in indices:
            mask[g, g] = True
        return mask

    keep_num = max(int(candidate_num * float(keep_ratio)), int(min_keep))
    keep_num = min(keep_num, candidate_num)
    threshold = torch.topk(score[candidate_mask], keep_num, largest=True).values[-1]
    gmask2d = (score >= threshold) & candidate_mask

    # These are part of the old QPM implementation and are shared by the
    # quality-component ablations to keep their candidate sets stable.
    for g in indices:
        gmask2d[g, g] = True
    gmask2d[min(indices), max(indices)] = True
    return gmask2d


def clips_to_adaptive_quality_gmask2d(
    glance_indices,
    glance_weights,
    num_clips,
    keep_ratio=0.5,
    min_keep=24,
    margin=3,
):
    return build_pseudo_gmask2d(
        glance_indices,
        glance_weights,
        num_clips,
        mode="qpm",
        keep_ratio=keep_ratio,
        min_keep=min_keep,
        margin=margin,
    )

if __name__ == "__main__":
    num_clips = 32
    duration = 29.42
    glance = 27.0 
    # gmask2d = glance_to_gmask2d(glance, num_clips, duration)
    anchors = [torch.randint(0, 16, (3,)),torch.randint(0, 16, (2,))]
    print(anchors)
    res = gen_gaussian_weight(anchors, 16, 0.3)
    # for r in res:
    #     print(r)
    print(res[0].size())
    
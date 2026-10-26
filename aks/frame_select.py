import heapq
import json
import numpy as np
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract pseudo keyframes with Original AKS or Modified AKS + TRS')

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ActivityNet-CD',
        help='support Charades-CD, ActivityNet-CD, Charades_STA and ActivityNet'
    )
    parser.add_argument(
        '--extract_feature_model',
        type=str,
        default='blip',
        help='blip/clip/sevila'
    )
    parser.add_argument(
        '--score_path',
        type=str,
        default='./outscores/ActivityNet-CD/blip/scores.json'
    )
    parser.add_argument(
        '--frame_path',
        type=str,
        default='./outscores/ActivityNet-CD/blip/frames.json'
    )
    parser.add_argument(
        '--video_sent',
        type=str,
        default='./datasets/ActivityNet-CD/anet_train.json'
    )
    parser.add_argument(
        '--max_num_frames',
        type=int,
        default=8
    )

    parser.add_argument(
        '--selector_mode',
        type=str,
        default='modified_aks_trs',
        choices=['original_aks', 'modified_aks_trs'],
        help=('original_aks: original depth-based AKS; '
              'modified_aks_trs: current modified AKS with RBA, mixed relevance, '
              'small-budget relaxation and Temporal Redundancy Suppression')
    )

    parser.add_argument(
        '--all_depth',
        type=int,
        default=3,
        help='maximum recursive depth used only by original_aks'
    )
    parser.add_argument(
        '--ratio',
        type=int,
        default=1
    )
    parser.add_argument(
        '--t1',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--t2',
        type=float,
        default=-100.0
    )
    parser.add_argument(
        '--relevance_alpha',
        type=float,
        default=0.9,
        help='R_B = alpha * MeanTopK(B) + (1-alpha) * Mean(B), alpha in [0, 1]'
    )
    parser.add_argument(
        '--trs_gap_ratio',
        type=float,
        default=0.6,
        help=(
            'Temporal Redundancy Suppression strength used only by modified_aks_trs. '
            'min_gap = round((region_length / budget) * trs_gap_ratio). '
            '0 disables TRS; 0.5 is a mild default.'
        )
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./selected_frames'
    )

    return parser.parse_args()


def normalize_weights(raw_scores):
    """
    将候选关键帧得分转换为权重列表（和帧位置一一对应）。
    使用稳定 softmax，保证权重和为 1。
    """
    if len(raw_scores) == 0:
        return []

    s = np.asarray(raw_scores, dtype=np.float32)
    s = s - np.max(s)
    e = np.exp(s)
    z = np.sum(e)

    if z <= 0 or np.isnan(z):
        return [1.0 / len(raw_scores)] * len(raw_scores)

    w = e / z
    return w.tolist()


def judge_region(score, judge_n, t1, t2):
    """
    保留原 AKS 的 Judge 思路：
    若当前区域中高分帧相对整体均值足够突出，则停止继续划分。

    注意：judge_n 始终使用全局 max_num_frames，
    不使用当前节点 budget，以尽量保持原 AKS 的判定方式不变。
    """
    if len(score) == 0:
        return True

    mean = np.mean(score)
    std = np.std(score)

    top_n = heapq.nlargest(
        min(judge_n, len(score)),
        range(len(score)),
        score.__getitem__
    )
    top_score = [score[t] for t in top_n]
    mean_diff = np.mean(top_score) - mean

    return mean_diff > t1 and std > t2



def original_aks_meanstd(dic_scores, fns, n, t1, t2, all_depth):
    """
    原始 AKS 的递归 Judge/Split：
    - 每个区域携带 depth；
    - Judge 不满足且 depth < all_depth 时，从时间中点二分；
    - 叶子区域最终按 floor(K / 2^depth) 分配关键帧名额。
    """
    split_scores = []
    split_fns = []
    no_split_scores = []
    no_split_fns = []

    for dic_score, fn in zip(dic_scores, fns):
        score = np.asarray(dic_score['score'], dtype=np.float32)
        depth = int(dic_score['depth'])
        fn = list(fn)

        if len(score) == 0:
            continue

        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(
            min(n, len(score)),
            range(len(score)),
            score.__getitem__
        )
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean

        if mean_diff > t1 and std > t2:
            no_split_scores.append({'score': score, 'depth': depth})
            no_split_fns.append(fn)

        elif depth < all_depth and len(score) > 1:
            mid = len(score) // 2

            score1 = score[:mid]
            score2 = score[mid:]
            fn1 = fn[:mid]
            fn2 = fn[mid:]

            if len(score1) > 0:
                split_scores.append({'score': score1, 'depth': depth + 1})
                split_fns.append(fn1)
            if len(score2) > 0:
                split_scores.append({'score': score2, 'depth': depth + 1})
                split_fns.append(fn2)
        else:
            no_split_scores.append({'score': score, 'depth': depth})
            no_split_fns.append(fn)

    if len(split_scores) > 0:
        child_scores, child_fns = original_aks_meanstd(
            split_scores,
            split_fns,
            n,
            t1,
            t2,
            all_depth
        )
        no_split_scores.extend(child_scores)
        no_split_fns.extend(child_fns)

    return no_split_scores, no_split_fns


def original_aks_select(score, fn, num, t1, t2, all_depth):
    """
    原始 AKS 选帧：叶子区域深度为 d 时，名额为 floor(K / 2^d)。
    返回 (score, frame_position) 列表。
    """
    leaves, leaf_fns = original_aks_meanstd(
        [{'score': np.asarray(score, dtype=np.float32), 'depth': 0}],
        [list(fn)],
        num,
        t1,
        t2,
        all_depth
    )

    candidates = []

    for leaf, leaf_fn in zip(leaves, leaf_fns):
        leaf_score = np.asarray(leaf['score'], dtype=np.float32)
        depth = int(leaf['depth'])

        leaf_k = int(num / (2 ** depth))
        if leaf_k <= 0 or len(leaf_score) == 0:
            continue

        topk = heapq.nlargest(
            min(leaf_k, len(leaf_score)),
            range(len(leaf_score)),
            leaf_score.__getitem__
        )

        for t in topk:
            candidates.append((float(leaf_score[t]), leaf_fn[t]))

    return candidates

def region_relevance(score, probe_k, alpha):
    """
    计算一个子区域 B 的 query relevance：

        R_B = alpha * MeanTopK(B) + (1 - alpha) * Mean(B)

    其中：
    - MeanTopK(B)：区域内 Top-k frame-query scores 的均值，
      强调局部高相关帧；
    - Mean(B)：区域内全部 frame-query scores 的均值，
      强调区域整体的持续相关性；
    - alpha ∈ [0, 1]：两者的平衡系数。

    当 alpha = 1 时，退化为原来的纯 MeanTopK relevance；
    当 alpha = 0 时，只使用区域整体均值。

    probe_k 仍由父节点当前 budget 决定：
        probe_k = max(1, budget // 2)
    """
    if len(score) == 0:
        return 0.0

    alpha = float(np.clip(alpha, 0.0, 1.0))

    k = min(max(1, probe_k), len(score))
    topk = heapq.nlargest(k, range(len(score)), score.__getitem__)

    mean_topk = float(np.mean([score[t] for t in topk]))
    mean_all = float(np.mean(score))

    relevance = alpha * mean_topk + (1.0 - alpha) * mean_all
    return float(relevance)


def allocate_relevance_budget(
    budget,
    left_relevance,
    right_relevance,
    left_capacity,
    right_capacity,
    eps=1e-12
):
    """
    Relevance-Aware Budget Allocation (RBA).

    给定当前节点预算 budget，将其分给左右两个子区域：
        K_L + K_R = budget

    规则：
    1) 根据左右区域 relevance 的比例分配；
    2) 只要左右区域都非空且 budget >= 2，则两侧至少各保留 1 个预算；
    3) 分配结果不能超过对应子区域实际帧数；
    4) 若某侧容量不足，剩余预算自动转移给另一侧。
    """
    total_capacity = left_capacity + right_capacity
    budget = min(budget, total_capacity)

    if budget <= 0:
        return 0, 0

    if left_capacity == 0:
        return 0, min(budget, right_capacity)
    if right_capacity == 0:
        return min(budget, left_capacity), 0

    # 只有 1 个预算时，正常流程不会进入这里；保留保护逻辑。
    if budget == 1:
        if left_relevance >= right_relevance:
            return 1, 0
        return 0, 1

    relevance_sum = left_relevance + right_relevance

    # 两侧 relevance 都接近 0 时退化为近似均分。
    if relevance_sum <= eps:
        left_budget = budget // 2
    else:
        p_left = left_relevance / relevance_sum
        # 避免 Python round 的 bankers rounding。
        left_budget = int(np.floor(budget * p_left + 0.5))

    # 放开“小预算两侧至少各 1 个”的硬约束：
    # 当父区域预算很小（例如 K<=4）时，允许更相关的一侧拿走全部预算，
    # 从而出现 2+0、4+0 这类分配。
    # 当预算更大时，仍保留两边至少 1 个预算，以维持基本的 temporal exploration。
    if budget <= 4:
        left_budget = max(0, min(left_budget, budget))
    else:
        left_budget = max(1, min(left_budget, budget - 1))

    right_budget = budget - left_budget

    # 容量约束：某侧分到的预算不能超过实际帧数。
    if left_budget > left_capacity:
        overflow = left_budget - left_capacity
        left_budget = left_capacity
        right_budget += overflow

    if right_budget > right_capacity:
        overflow = right_budget - right_capacity
        right_budget = right_capacity
        left_budget += overflow

    # 再做一次边界保护。
    left_budget = min(left_budget, left_capacity)
    right_budget = min(right_budget, right_capacity)

    # 理论上 total_capacity >= budget，因此这里应该仍可补足 budget。
    missing = budget - (left_budget + right_budget)
    if missing > 0:
        left_room = left_capacity - left_budget
        add_left = min(missing, left_room)
        left_budget += add_left
        missing -= add_left

    if missing > 0:
        right_room = right_capacity - right_budget
        add_right = min(missing, right_room)
        right_budget += add_right
        missing -= add_right

    return left_budget, right_budget


def select_topk(score, fn, budget):
    """
    在当前区域直接选择 top-budget 帧。
    返回 (score, frame_position) 列表。
    """
    if budget <= 0 or len(score) == 0:
        return []

    k = min(budget, len(score))
    topk = heapq.nlargest(k, range(len(score)), score.__getitem__)
    return [(float(score[t]), fn[t]) for t in topk]


def select_topk_temporal(score, fn, budget, gap_ratio=0.5):
    """
    Temporal Redundancy Suppression (TRS).

    与普通 top-k 不同，这里先按 score 从高到低扫描候选，
    只有当当前帧与已经选中的帧在“当前区域索引”上距离足够大时才保留。

    自适应最小间隔：
        ideal_spacing = len(region) / budget
        min_gap = round(ideal_spacing * gap_ratio)

    默认 gap_ratio=0.5：
    只要求达到理想均匀间距的一半，因此是较温和的去冗余，
    不会把选择器强制变成均匀采样。

    如果严格间隔约束下不足 budget 个，则按原始 score 从高到低补齐，
    保证最终仍输出 K 个关键帧。
    """
    if budget <= 0 or len(score) == 0:
        return []

    k = min(budget, len(score))
    if k <= 1 or gap_ratio <= 0:
        return select_topk(score, fn, k)

    ideal_spacing = len(score) / float(k)
    min_gap = max(1, int(np.floor(ideal_spacing * gap_ratio + 0.5)))

    ranked = heapq.nlargest(
        len(score),
        range(len(score)),
        score.__getitem__
    )

    selected_idx = []
    rejected_idx = []

    for idx in ranked:
        if all(abs(idx - s) >= min_gap for s in selected_idx):
            selected_idx.append(idx)
            if len(selected_idx) == k:
                break
        else:
            rejected_idx.append(idx)

    # 如果区域过短或分数峰值过密，严格 TRS 可能不足 K 个。
    # 此时按原始相关性补齐，避免改变关键帧总数。
    if len(selected_idx) < k:
        selected_set = set(selected_idx)
        for idx in ranked:
            if idx in selected_set:
                continue
            selected_idx.append(idx)
            selected_set.add(idx)
            if len(selected_idx) == k:
                break

    return [(float(score[t]), fn[t]) for t in selected_idx]


def adaptive_select(
    score,
    fn,
    budget,
    judge_n,
    t1,
    t2,
    relevance_alpha,
    use_trs=False,
    trs_gap_ratio=0.6
):
    """
    带 Relevance-Aware Budget Allocation 的递归关键帧选择。

    与原 AKS 的主要区别：
    - 原 AKS 的子区域预算依赖固定递归层级；
    - 这里取消 depth / all_depth，budget 由父节点显式传递，
      并依据左右子区域的混合 relevance 动态分配。

    停止条件：
    1) 当前区域为空 / 无预算；
    2) 当前区域帧数 <= 当前 budget；
    3) 当前 budget <= 1；
    4) 原 AKS Judge 条件满足。

    因为每次 split 后左右子区域至少各获得 1 个预算，
    所以继续递归的任一分支 budget 都会严格减小，
    最终必然触发 budget <= 1，无需额外最大深度参数。
    """
    score = np.asarray(score, dtype=np.float32)
    fn = list(fn)

    def stop_select(local_score, local_fn, local_budget):
        if use_trs:
            return select_topk_temporal(
                local_score,
                local_fn,
                local_budget,
                gap_ratio=trs_gap_ratio
            )
        return select_topk(local_score, local_fn, local_budget)

    if budget <= 0 or len(score) == 0:
        return []

    budget = min(budget, len(score))

    # 停止条件 1：候选数已经不超过预算。
    if len(score) <= budget:
        return stop_select(score, fn, budget)

    # 停止条件 2：只剩 1 个预算，无法再保证左右两侧都有预算。
    if budget <= 1:
        return stop_select(score, fn, 1)

    # 停止条件 3：保留原 AKS Judge。
    if judge_region(score, judge_n, t1, t2):
        return stop_select(score, fn, budget)

    # 防止极短区域继续划分。
    mid = len(score) // 2
    if mid <= 0 or mid >= len(score):
        return stop_select(score, fn, budget)

    left_score = score[:mid]
    right_score = score[mid:]
    left_fn = fn[:mid]
    right_fn = fn[mid:]

    # 用父节点当前 budget 的“近似半预算”确定 Top-k relevance 的观察尺度。
    probe_k = max(1, budget // 2)
    left_relevance = region_relevance(
        left_score, probe_k, relevance_alpha
    )
    right_relevance = region_relevance(
        right_score, probe_k, relevance_alpha
    )

    left_budget, right_budget = allocate_relevance_budget(
        budget=budget,
        left_relevance=left_relevance,
        right_relevance=right_relevance,
        left_capacity=len(left_score),
        right_capacity=len(right_score)
    )

    left_selected = adaptive_select(
        left_score,
        left_fn,
        left_budget,
        judge_n,
        t1,
        t2,
        relevance_alpha,
        use_trs,
        trs_gap_ratio
    )

    right_selected = adaptive_select(
        right_score,
        right_fn,
        right_budget,
        judge_n,
        t1,
        t2,
        relevance_alpha,
        use_trs,
        trs_gap_ratio
    )

    return left_selected + right_selected


def main(args):
    max_num_frames = args.max_num_frames
    ratio = args.ratio
    t1 = args.t1
    t2 = args.t2
    relevance_alpha = args.relevance_alpha
    trs_gap_ratio = args.trs_gap_ratio
    selector_mode = args.selector_mode
    all_depth = args.all_depth

    if selector_mode not in ('original_aks', 'modified_aks_trs'):
        raise ValueError(f'Internal selector_mode must be concrete, got {selector_mode}')
    if all_depth < 0:
        raise ValueError(f'--all_depth must be >= 0, got {all_depth}')

    if not 0.0 <= relevance_alpha <= 1.0:
        raise ValueError(
            f'--relevance_alpha must be in [0, 1], got {relevance_alpha}'
        )

    if trs_gap_ratio < 0.0:
        raise ValueError(
            f'--trs_gap_ratio must be >= 0, got {trs_gap_ratio}'
        )

    outs = []
    num_sent = []
    name_sent = []

    count = 0
    offset = 0

    out_frames_video = []
    out_weights_video = []

    # 与原版严格一致：仍读取扁平 scores.json / frames.json。
    with open(args.score_path) as f:
        itm_outs = json.load(f)

    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    with open(args.video_sent) as f:
        video_sent = json.load(f)

    os.makedirs(
        os.path.join(args.output_file, args.dataset_name),
        exist_ok=True
    )

    out_score_path = os.path.join(
        args.output_file,
        args.dataset_name,
        args.extract_feature_model,
        selector_mode
    )

    os.makedirs(out_score_path, exist_ok=True)

    for key, value in video_sent.items():
        num_sent.append(len(value['sentences']))
        name_sent.append(key)

    for idx, (itm_out, fn_out) in enumerate(zip(itm_outs, fn_outs)):
        nums = int(len(itm_out) / ratio)

        new_score = [
            itm_out[num * ratio]
            for num in range(nums)
        ]

        new_fnum = [
            fn_out[num * ratio]
            for num in range(nums)
        ]

        score = np.asarray(new_score, dtype=np.float32)
        fn = new_fnum
        num = max_num_frames

        sent_frames = []
        sent_weights = []

        # 多帧模式
        if num > 1:
            if len(score) >= num and (np.max(score) - np.min(score)) > 1e-12:
                # 与原版保持一致：
                # 先对当前 query 的 frame-query scores 做 min-max normalization。
                normalized_data = (
                    (score - np.min(score)) /
                    (np.max(score) - np.min(score))
                )

                if selector_mode == 'original_aks':
                    candidates = original_aks_select(
                        score=normalized_data,
                        fn=fn,
                        num=num,
                        t1=t1,
                        t2=t2,
                        all_depth=all_depth
                    )
                else:
                    candidates = adaptive_select(
                        score=normalized_data,
                        fn=fn,
                        budget=num,
                        judge_n=num,
                        t1=t1,
                        t2=t2,
                        relevance_alpha=relevance_alpha,
                        use_trs=(selector_mode == 'modified_aks_trs'),
                        trs_gap_ratio=trs_gap_ratio
                    )

                # 全局按 score 降序，便于后续 confidence 与 frame 一一对应。
                candidates.sort(key=lambda x: x[0], reverse=True)

                # frame-level 显式去重，作为鲁棒性保护。
                uniq = {}
                for sc, pos in candidates:
                    if pos not in uniq:
                        uniq[pos] = sc

                selected = sorted(
                    uniq.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:num]

                sent_frames = [pos for pos, _ in selected]
                raw_w = [sc for _, sc in selected]
                sent_weights = normalize_weights(raw_w)

            else:
                # 分数长度不足或全相等：保持原代码 fallback。
                k = min(num, len(score))
                topk = heapq.nlargest(
                    k,
                    range(len(score)),
                    score.__getitem__
                )

                sent_frames = [fn[t] for t in topk]
                raw_w = [float(score[t]) for t in topk]

                if len(raw_w) == 0:
                    sent_weights = []
                elif len(set(raw_w)) <= 1:
                    sent_weights = [1.0 / len(raw_w)] * len(raw_w)
                else:
                    sent_weights = normalize_weights(raw_w)

        # 单帧模式
        else:
            if len(score) > 0:
                top1 = heapq.nlargest(
                    1,
                    range(len(score)),
                    score.__getitem__
                )
                sent_frames = [fn[t] for t in top1]
                sent_weights = [1.0]
            else:
                sent_frames = []
                sent_weights = []

        out_frames_video.append(sent_frames)
        out_weights_video.append(sent_weights)

        # 完成一个视频：将该视频对应的多个 query 聚合输出。
        if count < len(num_sent) and (idx + 1) == (offset + num_sent[count]):
            outs.append(dict(
                vid=name_sent[count],
                value=out_frames_video,
                weights=out_weights_video
            ))

            offset += num_sent[count]
            count += 1
            out_frames_video = []
            out_weights_video = []

    # 为消融实验区分 selector 和 K，避免互相覆盖。
    score_path = os.path.join(
        out_score_path,
        f'selected_frames_{max_num_frames}.json'
    )

    with open(score_path, 'w') as f:
        json.dump(outs, f)

    print('==================================================')
    print(f'Selector : {selector_mode}')
    print(f'Dataset  : {args.dataset_name}')
    print(f'K        : {max_num_frames}')
    if selector_mode == 'original_aks':
        print(f'Depth    : {all_depth}')
    else:
        print(f'Alpha    : {relevance_alpha}')
        if selector_mode == 'modified_aks_trs':
            print(f'TRS gap  : {trs_gap_ratio}')
    print(f'Saved to : {score_path}')
    print('==================================================')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)




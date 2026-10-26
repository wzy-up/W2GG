import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_module("old_default_utils", "w2gg/data/datasets/utils.py")
common = load_module("old_default_common", "w2gg/modeling/w2gg/common.py")


def test_full_preprocessing_uses_old_operation_order():
    indices, weights, _ = utils.prepare_pseudo_glances(
        raw_glance=[10, 11, 40, 75],
        raw_weights=[0.20, 0.30, 0.40, 0.10],
        fps=10,
        num_clips=16,
        duration=10,
        keyframe_mode="multi",
        max_keyframes=0,
        refinement="dedup_filter",
        weight_mode="confidence",
    )
    assert indices == [2, 6]
    assert weights.shape == (2,)
    assert torch.isclose(weights.sum(), torch.tensor(0.999999), atol=2e-6)


def test_full_qpm_keeps_old_forced_anchors():
    mask = utils.build_pseudo_gmask2d(
        [2, 6],
        torch.tensor([0.55, 0.45]),
        16,
        mode="qpm",
        keep_ratio=0.42,
        min_keep=24,
        margin=3,
    )
    assert mask.dtype == torch.bool
    assert mask[2, 2]
    assert mask[6, 6]
    assert mask[2, 6]


def test_dynamic_gaussian_uses_old_repeat_sum_result():
    counters = [torch.tensor([[2, 0, 2, 0]], dtype=torch.long)]
    sims = [torch.tensor([[1.0, 0.2, 0.9, 0.1]])]
    output = common.gen_multi_glances_gaussian_weight(
        counters, sims, 4, 0.3, cnum=1
    )
    assert output[0].shape == (1, 4, 4)
    assert torch.isfinite(output[0]).all()

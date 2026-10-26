import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_module("ablation_utils", "w2gg/data/datasets/utils.py")
common = load_module("ablation_common", "w2gg/modeling/w2gg/common.py")


def test_prepare_refinement_merges_and_filters():
    indices, weights, primary = utils.prepare_pseudo_glances(
        raw_glance=[20, 22, 50, 90],
        raw_weights=[0.30, 0.25, 0.35, 0.10],
        fps=10,
        num_clips=10,
        duration=10,
        refinement="dedup_filter",
    )
    assert len(indices) == len(weights)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert len(indices) <= 3
    assert isinstance(primary, int)


def test_single_and_union_masks_are_distinct():
    indices = [2, 6]
    weights = torch.tensor([0.8, 0.2])
    single = utils.build_pseudo_gmask2d(indices, weights, 8, mode="single")
    union = utils.build_pseudo_gmask2d(indices, weights, 8, mode="union")
    assert single[0, 2]
    assert not single[5, 6]
    assert union[5, 6]
    assert union.sum() > single.sum()


def test_qpm_keeps_minimum_span():
    indices = [2, 5]
    weights = torch.tensor([0.6, 0.4])
    qpm = utils.build_pseudo_gmask2d(
        indices, weights, 8, mode="qpm", keep_ratio=0.25, min_keep=1, margin=1
    )
    assert qpm[2, 5]
    assert qpm.dtype == torch.bool


def test_static_multi_gaussian_shape():
    glances = [[torch.tensor([2, 5])]]
    weights = [[torch.tensor([0.7, 0.3])]]
    output = common.gen_static_multi_glances_gaussian_weight(glances, weights, 8, 0.3)
    assert len(output) == 1
    assert output[0].shape == (1, 8, 8)
    assert torch.isfinite(output[0]).all()


def test_dynamic_gaussian_shape():
    counters = [torch.tensor([[2, 0, 2, 0]], dtype=torch.long)]
    sims = [torch.tensor([[1.0, 0.2, 0.9, 0.1]])]
    output = common.gen_multi_glances_gaussian_weight(counters, sims, 4, 0.3, cnum=2)
    assert output[0].shape == (1, 4, 4)
    assert torch.isfinite(output[0]).all()


class FakeBatch:
    def __init__(self):
        self.glances = [[torch.tensor([2, 5])]]
        self.glance_weights = [[torch.tensor([0.4, 0.6])]]
        # Original top raw frame is kept separately and may differ after refinement.
        self.primary_glances = [torch.tensor([2])]


def test_single_anchor_memory_uses_primary_glance():
    loader = [(FakeBatch(), (7,))]
    memory = common.init_glance_memory(
        loader, length=8, dga_mode="single_anchor", continuous_num=2
    )
    assert memory[7][0, 2] == 3
    assert memory[7][0, 5] == 0


def test_multi_weighted_memory_uses_refined_glances():
    loader = [(FakeBatch(), (7,))]
    memory = common.init_glance_memory(
        loader, length=8, dga_mode="multi_weighted", continuous_num=2
    )
    assert memory[7][0, 2] >= 2
    assert memory[7][0, 5] >= 2

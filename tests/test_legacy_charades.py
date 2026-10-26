"""Offline regression tests for the legacy Charades-STA data path."""
import json
import logging
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from w2gg.config import cfg
from w2gg.config.paths_catalog import DatasetCatalog
from w2gg.data import build_dataset
from w2gg.data.collate_batch import BatchCollator
from w2gg.data.datasets.charades import CharadesDataset
from w2gg.data.datasets.evaluation import evaluate
from w2gg.modeling.w2gg.featpool import build_featpool
import test_net


ROOT = Path(__file__).resolve().parents[1]


class OfflineTokenizer:
    def __call__(self, sentences, **kwargs):
        shape = (len(sentences), 4)
        return {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


class OfflineModel(torch.nn.Module):
    """Small checkpoint-backed model to exercise the real evaluation entry."""
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.ones(16, 16).triu())

    def forward(self, batch):
        return [self.scores.unsqueeze(0).expand(count, -1, -1) for count in batch.num_sentence]


class LegacyCharadesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = patch(
            "w2gg.data.datasets.charades.DistilBertTokenizer.from_pretrained",
            return_value=OfflineTokenizer(),
        )
        self.tokenizer.start()
        self.addCleanup(self.tokenizer.stop)
        self.config = cfg.clone()
        self.config.merge_from_file(str(ROOT / "configs/pool_charades_16x16_k5l8.yaml"))

    def write_json(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
        return str(path)

    def annotation(self, duration_key="duration"):
        return {duration_key: 10.0, "timestamps": [[1, 5]], "sentences": ["a person sits."]}

    def dataset(self, ann_file, **kwargs):
        return CharadesDataset(
            ann_file, str(self.root / "features.hdf5"),
            num_pre_clips=32, num_clips=16, **kwargs,
        )

    def test_full_legacy_test_split_needs_no_training_files(self):
        path = ROOT / "旧版标注文件/Charades_STA/charades_test.json"
        annotations = json.loads(path.read_text(encoding="utf-8"))
        dataset = self.dataset(
            str(path), selected_frames="nonexistent.json",
            selected_fps_file="nonexistent_fps.json",
        )
        self.assertEqual(len(dataset), 1334)
        self.assertEqual([dataset.get_vid(i) for i in range(len(dataset))], list(annotations))
        self.assertEqual(
            sum(len(item["sentence"]) for item in dataset.annos),
            sum(len(item["sentences"]) for item in annotations.values()),
        )
        self.assertAlmostEqual(dataset.get_duration(0), 30.96)
        self.assertFalse(dataset.annos[0]["gmasks2d"].any())

    def test_new_duration_schema_remains_supported(self):
        path = self.write_json("test.json", {"new": self.annotation("video_duration")})
        self.assertEqual(self.dataset(path).get_duration(0), 10.0)

    def test_training_keyframes_are_matched_by_video_id(self):
        ann_file = self.write_json("train.json", {"a": self.annotation(), "b": self.annotation()})
        frames = self.write_json("frames.json", [
            {"vid": "b", "value": [[80]], "weights": [[1.0]]},
            {"vid": "a", "value": [[20]], "weights": [[1.0]]},
        ])
        fps = self.write_json("fps.json", {"a": 10, "b": 10})
        dataset = self.dataset(ann_file, selected_frames=frames, selected_fps_file=fps)
        self.assertEqual(dataset.annos[0]["glances"][0].tolist(), [3])
        self.assertEqual(dataset.annos[1]["glances"][0].tolist(), [12])

    def test_missing_training_video_cannot_silently_truncate_dataset(self):
        ann_file = self.write_json("train.json", {"a": self.annotation(), "b": self.annotation()})
        frames = self.write_json("frames.json", [{"vid": "a", "value": [[20]]}])
        fps = self.write_json("fps.json", {"a": 10, "b": 10})
        with self.assertRaisesRegex(ValueError, "Missing Charades keyframes"):
            self.dataset(ann_file, selected_frames=frames, selected_fps_file=fps)

    def test_catalog_and_periodic_validation_use_legacy_test_split(self):
        self.write_json("Charades_STA/charades_test.json", {"a": self.annotation()})
        # A training selector choice must have no effect on evaluation.
        self.config.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY = "unavailable"
        with patch.object(DatasetCatalog, "DATA_DIR", str(self.root)):
            args = DatasetCatalog.get("charades_test")["args"]
            self.assertTrue(args["feat_file"].endswith("vgg_rgb_features.hdf5"))
            self.assertNotIn("selected_frames", args)
            for periodic in (False, True):
                dataset = build_dataset(
                    self.config.DATASETS.TEST, DatasetCatalog, self.config, is_train=periodic,
                )[0]
                self.assertEqual(len(dataset), 1)

    def test_vgg_features_collate_and_pass_feature_pool(self):
        ann_file = self.write_json("test.json", {"a": self.annotation(), "b": self.annotation()})
        with h5py.File(self.root / "features.hdf5", "w") as handle:
            for vid in ("a", "b"):
                handle[vid] = np.ones((40, 4096), dtype=np.float32)
        dataset = self.dataset(ann_file, expected_feature_size=4096)
        batch, indices = BatchCollator()([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch.feats.shape), (2, 32, 4096))
        self.assertEqual(indices, (0, 1))
        with torch.no_grad():
            output = build_featpool(self.config)(batch.feats)
        self.assertEqual(tuple(output.shape), (2, 512, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_incompatible_feature_dimension_has_clear_error(self):
        ann_file = self.write_json("test.json", {"a": self.annotation()})
        with h5py.File(self.root / "features.hdf5", "w") as handle:
            handle["a"] = np.ones((40, 1024), dtype=np.float32)
        dataset = self.dataset(ann_file, expected_feature_size=4096)
        with self.assertRaisesRegex(ValueError, "INPUT_SIZE is 4096"):
            dataset[0]

    def test_checkpoint_accepts_training_ddp_and_plain_state_dict(self):
        source = torch.nn.Linear(4, 2)
        for content in (
            source.state_dict(),
            {"model": source.state_dict(), "epoch": 3},
            {"model": {"module." + key: value for key, value in source.state_dict().items()}},
        ):
            path = self.root / "model.pth"
            torch.save(content, path)
            target = torch.nn.Linear(4, 2)
            test_net.load_test_checkpoint(target, str(path))
            self.assertTrue(torch.equal(source.weight, target.weight))
            self.assertTrue(torch.equal(source.bias, target.bias))

    def test_checkpoint_rejects_missing_weights_and_wrong_dimensions(self):
        path = self.root / "bad.pth"
        for content in ({"weight": torch.ones(2, 4)}, torch.nn.Linear(3, 2).state_dict()):
            torch.save(content, path)
            with self.assertRaisesRegex(RuntimeError, "Checkpoint does not match"):
                test_net.load_test_checkpoint(torch.nn.Linear(4, 2), str(path))

    def test_test_entry_runs_legacy_loader_and_real_evaluation(self):
        self.write_json("Charades_STA/charades_test.json", {"a": self.annotation()})
        with h5py.File(self.root / "Charades_STA/vgg_rgb_features.hdf5", "w") as handle:
            handle["a"] = np.ones((40, 4096), dtype=np.float32)
        checkpoint = self.root / "model.pth"
        torch.save({"model": OfflineModel().state_dict()}, checkpoint)
        logger = logging.getLogger("legacy_entry_test")
        args = [
            "test_net.py", "--data-dir", str(self.root), "--ckpt", str(checkpoint),
            "MODEL.DEVICE", "cpu", "TEST.BATCH_SIZE", "1",
            "OUTPUT_DIR", str(self.root / "output"),
        ]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch.object(test_net, "cfg", cfg.clone()), patch.object(test_net, "setup_logger", return_value=logger):
                with patch.object(test_net, "build_model", return_value=OfflineModel()):
                    with patch.object(test_net, "inference", wraps=test_net.inference) as run_inference:
                        test_net.main()
                        self.assertEqual(run_inference.call_count, 1)
                        self.assertEqual(run_inference.call_args.kwargs["dataset_name"], "charades_test")
                        dataset = run_inference.call_args.args[2].dataset
                        self.assertEqual(len(dataset), 1)
                        self.assertEqual(dataset.get_vid(0), "a")


class RecallEvaluationTests(unittest.TestCase):
    def evaluate_scores(self, scores, moments, name="charades", ranks=(1, 5)):
        config = cfg.clone()
        config.DATASETS.NAME = name

        class Dataset:
            def __len__(self):
                return 1

            def get_duration(self, idx):
                return 10.0

            def get_moment(self, idx):
                return torch.tensor(moments, dtype=torch.float32)

        return evaluate(
            config, Dataset(), [{"contrastive": scores}],
            nms_thresh=0.5, recall_metrics=ranks,
        )

    def test_recall_counts_threshold_equality_without_boundary_discount(self):
        scores = torch.zeros(1, 10, 10)
        scores[0, 0, 7] = 0.9  # [0, 8] has IoU exactly 0.5 with [0, 4].
        scores[0, 0, 3] = 0.8  # Rank 2 is exact; NMS keeps IoU == threshold.
        for position in (1, 4, 8):
            scores[0, position, position] = 0.1  # Five candidates survive NMS.
        result = self.evaluate_scores(scores, [[0, 4]])
        expected = {
            "R@1,IoU@0.5": 1.0, "R@1,IoU@0.7": 0.0,
            "R@5,IoU@0.5": 1.0, "R@5,IoU@0.7": 1.0,
            "Avg_R1": 0.5, "Avg_R5": 1.0,
        }
        self.assertEqual(set(result), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(float(result[key]), value)

    def test_original_expand_behavior_is_preserved(self):
        # The supplied implementation raises for 0 or 2 candidates at R@5.
        for candidate_count in (0, 2):
            scores = torch.zeros(1, 10, 10)
            for position in range(candidate_count):
                scores[0, position, position] = 1.0
            with self.assertRaises(RuntimeError):
                self.evaluate_scores(scores, [[0, 4]])

    def test_activitynet_and_tacos_use_requested_thresholds(self):
        scores = torch.zeros(1, 10, 10)
        scores[0, 0, 9] = 1.0  # IoU 0.4: only the 0.3 threshold is a hit.
        for name in ("activitynet", "tacos"):
            with self.subTest(dataset=name):
                result = self.evaluate_scores(scores, [[0, 4]], name=name)
                for rank in (1, 5):
                    self.assertEqual(float(result["R@{},IoU@0.3".format(rank)]), 1.0)
                    self.assertEqual(float(result["R@{},IoU@0.5".format(rank)]), 0.0)
                    self.assertEqual(float(result["R@{},IoU@0.7".format(rank)]), 0.0)
                    self.assertAlmostEqual(float(result["Avg_R{}".format(rank)]), 1 / 3)

    def test_seven_tenths_iou_is_a_hit(self):
        scores = torch.zeros(1, 10, 10)
        scores[0, 0, 9] = 1.0
        result = self.evaluate_scores(scores, [[0, 7]])
        self.assertEqual(float(result["R@1,IoU@0.7"]), 1.0)


if __name__ == "__main__":
    unittest.main()

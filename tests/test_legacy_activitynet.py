"""Offline checks for legacy ActivityNet annotations, selectors and HDF5 layouts."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from w2gg.config import cfg
from w2gg.config.paths_catalog import DatasetCatalog
from w2gg.data import make_data_loader
from w2gg.data.collate_batch import BatchCollator
from w2gg.data.datasets.activitynet import ActivityNetDataset
from w2gg.modeling.w2gg.featpool import build_featpool


ROOT = Path(__file__).resolve().parents[1]


class OfflineTokenizer:
    def __call__(self, sentences, **kwargs):
        tokens = torch.ones(len(sentences), 4, dtype=torch.long)
        return {"input_ids": tokens, "attention_mask": tokens}


class LegacyActivityNetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.feat = self.root / "features.hdf5"
        self.config = cfg.clone()
        self.config.merge_from_file(str(ROOT / "configs/pool_activitynet_64x64_k9l4.yaml"))
        self.config.DATALOADER.NUM_WORKERS = 0
        tokenizer = patch(
            "w2gg.data.datasets.activitynet.DistilBertTokenizer.from_pretrained",
            return_value=OfflineTokenizer(),
        )
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def write_json(self, path, content):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
        return str(path)

    def annotation(self):
        return {"duration": 10.0, "timestamps": [[1, 5]], "sentences": ["a person runs"]}

    def dataset(self, path, **kwargs):
        return ActivityNetDataset(path, str(self.feat), 256, 64, **kwargs)

    def test_supplied_training_selectors_cover_annotations_by_id(self):
        annotations = json.loads((ROOT / "旧版标注文件/ActivityNet/glance_train.json").read_text(encoding="utf-8"))
        items = json.loads((ROOT / "旧版关键帧/Activity/selected_frames.json").read_text(encoding="utf-8"))
        fps = json.loads((ROOT / "旧版关键帧/Activity/selected_fps.json").read_text(encoding="utf-8"))
        selected = {item["vid"]: item for item in items}
        self.assertEqual(len(annotations), 10009)
        self.assertEqual(sum(len(a["sentences"]) for a in annotations.values()), 37421)
        self.assertEqual(len(items), len(selected))
        self.assertEqual(set(annotations), set(selected))
        self.assertEqual(set(annotations), set(fps))
        for vid, anno in annotations.items():
            self.assertEqual(len(selected[vid]["value"]), len(anno["sentences"]))
            for frames in selected[vid]["value"]:
                self.assertTrue(all(0 <= frame / fps[vid] <= anno["duration"] for frame in frames))
        # Run real preprocessing on a small sample of the supplied selectors.
        sample = dict(list(annotations.items())[:3])
        path = self.write_json("train.json", sample)
        result = self.dataset(
            path,
            selected_frames=str(ROOT / "旧版关键帧/Activity/selected_frames.json"),
            selected_fps_file=str(ROOT / "旧版关键帧/Activity/selected_fps.json"),
        )
        self.assertEqual([result.get_vid(i) for i in range(len(result))], list(sample))

    def test_full_validation_and_test_do_not_read_training_files(self):
        for split, videos, queries in (("val", 4917, 17505), ("test", 4885, 17031)):
            with self.subTest(split=split):
                dataset = self.dataset(
                    str(ROOT / "旧版标注文件/ActivityNet" / (split + ".json")),
                    selected_frames="missing.json", selected_fps_file="missing_fps.json",
                )
                self.assertEqual(len(dataset), videos)
                self.assertEqual(sum(len(a["sentence"]) for a in dataset.annos), queries)
                self.assertFalse(dataset.annos[0]["gmasks2d"].any())

    def test_shuffled_selectors_follow_video_ids(self):
        ann = self.write_json("train.json", {"a": self.annotation(), "b": self.annotation()})
        selected = self.write_json("frames.json", [
            {"vid": "b", "value": [[80]], "weights": [[1.0]]},
            {"vid": "a", "value": [[20]], "weights": [[1.0]]},
        ])
        fps = self.write_json("fps.json", {"a": 10, "b": 10})
        dataset = self.dataset(ann, selected_frames=selected, selected_fps_file=fps)
        self.assertEqual(dataset.annos[0]["glances"][0].tolist(), [13])
        self.assertEqual(dataset.annos[1]["glances"][0].tolist(), [50])

    def test_missing_training_selector_fails_instead_of_dropping_video(self):
        ann = self.write_json("train.json", {"a": self.annotation(), "b": self.annotation()})
        selected = self.write_json("frames.json", [{"vid": "a", "value": [[20]], "weights": [[1.0]]}])
        fps = self.write_json("fps.json", {"a": 10, "b": 10})
        with self.assertRaisesRegex(ValueError, "Missing ActivityNet training data"):
            self.dataset(ann, selected_frames=selected, selected_fps_file=fps)

    def test_flat_and_grouped_c3d_features_match_and_feed_model(self):
        ann = self.write_json("test.json", {"flat": self.annotation(), "group": self.annotation()})
        array = np.ones((40, 500), dtype=np.float32)
        with h5py.File(self.feat, "w") as handle:
            handle["flat"] = array
            handle.create_group("group")["c3d_features"] = array
        dataset = self.dataset(ann, expected_feature_size=500)
        self.assertTrue(torch.equal(dataset[0][0], dataset[1][0]))
        batch, _ = BatchCollator()([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch.feats.shape), (2, 256, 500))
        with torch.no_grad():
            output = build_featpool(self.config)(batch.feats)
        self.assertEqual(tuple(output.shape), (2, 512, 64))

    def test_wrong_feature_dimension_is_reported(self):
        ann = self.write_json("test.json", {"a": self.annotation()})
        with h5py.File(self.feat, "w") as handle:
            handle["a"] = np.ones((5, 1024), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "INPUT_SIZE is 500"):
            self.dataset(ann, expected_feature_size=500)[0]

    def test_periodic_validation_and_final_test_use_distinct_annotations(self):
        self.write_json("ActivityNet/val.json", {"validation_video": self.annotation()})
        self.write_json("ActivityNet/test.json", {"test_video": self.annotation()})
        self.config.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY = "not_available"
        with patch.dict(os.environ, {"W2GG_DATA_DIR": str(self.root)}):
            val = make_data_loader(self.config, is_train=False, is_for_period=True)
            test = make_data_loader(self.config, is_train=False)[0]
        self.assertEqual(val.dataset.get_vid(0), "validation_video")
        self.assertEqual(test.dataset.get_vid(0), "test_video")

    def test_catalog_feature_and_selector_overrides(self):
        with patch.object(DatasetCatalog, "DATA_DIR", str(self.root)):
            with patch.dict(os.environ, {
                "W2GG_ACTIVITYNET_FEATURES": "features.hdf5",
                "W2GG_ACTIVITYNET_SELECTED_FRAMES": "selectors.json",
            }):
                train = DatasetCatalog.get("activitynet_train")["args"]
                test = DatasetCatalog.get("activitynet_test")["args"]
                self.assertEqual(train["feat_file"], str(self.feat))
                self.assertEqual(test["feat_file"], str(self.feat))
                self.assertEqual(train["selected_frames"], str(self.root / "selectors.json"))
                self.assertNotIn("selected_frames", test)
                self.assertTrue(train["ann_file"].endswith("ActivityNet/glance_train.json".replace("/", os.sep)) or train["ann_file"].endswith("ActivityNet/glance_train.json"))


if __name__ == "__main__":
    unittest.main()

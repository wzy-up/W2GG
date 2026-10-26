import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)



class DatasetCatalog(object):
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")

    ACTIVITYNET_SELECTED_FRAMES = os.path.abspath(
        os.path.expanduser(
            os.environ.get(
                "W2GG_ACTIVITYNET_SELECTED_FRAMES",
                os.path.join(
                    DATA_DIR,
                    "ActivityNet-CD/selected_frames.json"
                )
            )
        )
    )


    KEYFRAME_FILES = {
        "charades": {
            "rba": {
                "k1": "Charades_CD/selected_frames_1.json",
                "k2": "Charades_CD/selected_frames_2.json",
                "k4": "Charades_CD/selected_frames_4.json",
                "k6": "Charades_CD/selected_frames_6.json",
                "k8": "Charades_CD/selected_frames_8.json",
                "k10": "Charades_CD/selected_frames_10.json",
                "k12": "Charades_CD/selected_frames_12.json",
                "k16": "Charades_CD/selected_frames_16.json",
            },
            "original_aks": {
                "k1": "Charades_CD/original_aks/selected_frames_1.json",
                "k2": "Charades_CD/original_aks/selected_frames_2.json",
                "k4": "Charades_CD/original_aks/selected_frames_4.json",
                "k6": "Charades_CD/original_aks/selected_frames_6.json",
                "k8": "Charades_CD/original_aks/selected_frames_8.json",
                "k10": "Charades_CD/original_aks/selected_frames_10.json",
                "k12": "Charades_CD/original_aks/selected_frames_12.json",
                "k16": "Charades_CD/original_aks/selected_frames_16.json",
            },
        },
        "activitynet": {
            "rba": {
                "k1": "ActivityNet-CD/selected_frames_1.json",
                "k2": "ActivityNet-CD/selected_frames_2.json",
                "k4": "ActivityNet-CD/selected_frames_4.json",
                "k6": "ActivityNet-CD/selected_frames_6.json",
                "k8": "ActivityNet-CD/selected_frames_8.json",
                "k10": "ActivityNet-CD/selected_frames_10.json",
                "k12": "ActivityNet-CD/selected_frames_12.json",
                "k16": "ActivityNet-CD/selected_frames_16.json",
            },
            "original_aks": {
                "k1": "ActivityNet-CD/original_aks/selected_frames_1.json",
                "k2": "ActivityNet-CD/original_aks/selected_frames_2.json",
                "k4": "ActivityNet-CD/original_aks/selected_frames_4.json",
                "k6": "ActivityNet-CD/original_aks/selected_frames_6.json",
                "k8": "ActivityNet-CD/original_aks/selected_frames_8.json",
                "k10": "ActivityNet-CD/original_aks/selected_frames_10.json",
                "k12": "ActivityNet-CD/original_aks/selected_frames_12.json",
                "k16": "ActivityNet-CD/original_aks/selected_frames_16.json",
            },
        },
    }

    DATASETS = {
        "activitynet_train": {
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet-CD/anet_train.json",
            "feat_file": "ActivityNet-CD/I3D_ORG.hdf5",
            "selected_frames": "ActivityNet-CD/selected_frames_2.json",
            "selected_fps_file": "ActivityNet-CD/selected_fps.json",
        },
        "activitynet_val": {
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet-CD/anet_val.json",
            "feat_file": "ActivityNet-CD/I3D_ORG.hdf5",
            "selected_frames": "ActivityNet-CD/selected_frames_2.json",
            "selected_fps_file": "ActivityNet-CD/selected_fps.json",
        },
        "activitynet_test": {
            "video_dir": "ActivityNet/videos",
            "ann_file": "ActivityNet-CD/anet_test_ood.json",
            "feat_file": "ActivityNet-CD/I3D_ORG.hdf5",
            "selected_frames": "ActivityNet-CD/selected_frames_2.json",
            "selected_fps_file": "ActivityNet-CD/selected_fps.json",
        },
        "charades_train": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_CD/charades_train.json",
            "feat_file": "Charades_CD/I3D_NEW.hdf5",
            "selected_frames": "Charades_CD/selected_frames_4.json",
            "selected_fps_file": "Charades_CD/selected_fps.json",
        },
        "charades_test": {
            "video_dir": "Charades_STA/videos",
            "ann_file": "Charades_CD/charades_test_ood.json",
            "feat_file": "Charades_CD/I3D_NEW.hdf5",
            "selected_frames": "Charades_CD/selected_frames_4.json",
            "selected_fps_file": "Charades_CD/selected_fps.json",
        },
    }

    @staticmethod
    def _family(name):
        if "activitynet" in name:
            return "activitynet"
        if "charades" in name:
            return "charades"
        raise RuntimeError("Dataset not available: {}".format(name))

    @staticmethod
    def _normalize_selector(selector):
        """Normalize convenient selector aliases to canonical names."""
        selector = str(selector or "rba").strip().lower()
        aliases = {
            "rba": "rba",
            "rab": "rba",
            "modified": "rba",
            "modified_aks": "rba",
            "original": "original_aks",
            "aks": "original_aks",
            "original_aks": "original_aks",
        }
        if selector not in aliases:
            raise KeyError(
                "Unknown pseudo-keyframe selector={!r}. Available selectors: "
                "original_aks, rba (alias: rab).".format(selector)
            )
        return aliases[selector]

    @staticmethod
    def get_keyframe_file(name, key, selector="rba"):
        """Resolve selected_frames by dataset family, selector family, and K key."""
        family = DatasetCatalog._family(name)
        key = str(key)
        selector = DatasetCatalog._normalize_selector(selector)

        family_table = DatasetCatalog.KEYFRAME_FILES.get(family, {})
        table = family_table.get(selector, {})
        if key not in table:
            available = ", ".join(sorted(table)) or "<none>"
            raise KeyError(
                "No pseudo-keyframe file configured for dataset={!r}, "
                "selector={!r}, key={!r}. Available keys: {}. Edit "
                "DatasetCatalog.KEYFRAME_FILES in paths_catalog.py.".format(
                    family, selector, key, available
                )
            )
        return os.path.join(DatasetCatalog.DATA_DIR, table[key])

    @staticmethod
    def _resolve(path):
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return path
        return os.path.join(DatasetCatalog.DATA_DIR, path)

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            ann_file=DatasetCatalog._resolve(attrs["ann_file"]),
            feat_file=DatasetCatalog._resolve(attrs["feat_file"]),
        )
        if "selected_frames" in attrs:
            args["selected_frames"] = DatasetCatalog._resolve(attrs["selected_frames"])
        if "selected_fps_file" in attrs:
            args["selected_fps_file"] = DatasetCatalog._resolve(attrs["selected_fps_file"])

        family = DatasetCatalog._family(name)
        factories = {
            "tacos": "TACoSDataset",
            "activitynet": "ActivityNetDataset",
            "charades": "CharadesDataset",
        }
        return {"factory": factories[family], "args": args}

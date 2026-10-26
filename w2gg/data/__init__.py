import os
import logging
import torch
from w2gg.utils.comm import get_world_size
from w2gg.utils.imports import import_file
from . import datasets as D
from .samplers import DistributedSampler
from .collate_batch import BatchCollator

def build_dataset(dataset_list, dataset_catalog, cfg, is_train=True):
    # build specific dataset
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(
                dataset_list
            )
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"].copy()
        # Resolve the independently generated selector output. An explicit file
        # override has priority; otherwise a short key (k1/k4/k16/...) is looked
        # up in DatasetCatalog.KEYFRAME_FILES.
        selected_frames_override = cfg.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_FILE
        selected_frames_key = cfg.DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY
        selected_frames_selector = cfg.DATASETS.PSEUDO_GLANCE.SELECTOR
        if selected_frames_override:
            selected_frames_override = os.path.expanduser(str(selected_frames_override))
            if not os.path.isabs(selected_frames_override):
                selected_frames_override = os.path.join(
                    dataset_catalog.DATA_DIR, selected_frames_override
                )
            args["selected_frames"] = selected_frames_override
        elif selected_frames_key:
            args["selected_frames"] = dataset_catalog.get_keyframe_file(
                dataset_name, selected_frames_key, selected_frames_selector
            )
        args["num_pre_clips"] = cfg.INPUT.NUM_PRE_CLIPS
        args["num_clips"] = cfg.MODEL.W2GG.NUM_CLIPS
        # Pseudo-glance and proposal-mask switches shared by all datasets.
        args["pseudo_keyframe_mode"] = cfg.DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE
        args["pseudo_max_keyframes"] = cfg.DATASETS.PSEUDO_GLANCE.MAX_KEYFRAMES
        args["pseudo_refinement"] = cfg.DATASETS.PSEUDO_GLANCE.REFINEMENT
        args["pseudo_filter_threshold_scale"] = cfg.DATASETS.PSEUDO_GLANCE.FILTER_THRESHOLD_SCALE
        args["pseudo_weight_mode"] = cfg.DATASETS.PSEUDO_GLANCE.WEIGHT_MODE
        args["gmask_mode"] = cfg.DATASETS.PSEUDO_GMASK.MODE
        args["gmask_keep_ratio"] = cfg.DATASETS.PSEUDO_GMASK.KEEP_RATIO
        args["gmask_min_keep"] = cfg.DATASETS.PSEUDO_GMASK.MIN_KEEP
        args["gmask_margin"] = cfg.DATASETS.PSEUDO_GMASK.MARGIN
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)
    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_train_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
        # TODO: check if drop_last=True helps
    )
    return batch_sampler

def make_test_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, is_for_period=False):
    num_gpus = get_world_size()
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        assert (
            batch_size % num_gpus == 0
        ), "SOLVER.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
            batch_size, num_gpus)
        batch_size_per_gpu = batch_size // num_gpus
        shuffle = True
        max_epoch = cfg.SOLVER.MAX_EPOCH
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        assert (
            batch_size % num_gpus == 0
        ), "TEST.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
            batch_size, num_gpus)
        batch_size_per_gpu = batch_size // num_gpus
        shuffle = True if not is_distributed else False  # originally False

    if batch_size_per_gpu > 1:
        logger = logging.getLogger(__name__)

    paths_catalog = import_file(
        "w2gg.cfg.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, DatasetCatalog, cfg, is_train=is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        if is_train:
            batch_sampler = make_train_data_sampler(dataset, sampler, batch_size_per_gpu)
        else:
            batch_sampler = make_test_data_sampler(dataset, sampler, batch_size_per_gpu)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(),
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

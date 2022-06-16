"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import pickle
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch.utils.data
import torch.utils.data as data
from detectron2.config import configurable
from detectron2.data import DatasetCatalog
from detectron2.data.build import build_batch_data_loader
from detectron2.data.build import (
    filter_images_with_only_crowd_annotations,
    check_metadata_consistency,
)
from detectron2.data.common import (
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    TrainingSampler,
)
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import _log_api_usage
from sylph.data.utils import temp_seed

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_meta_detection_train_loader",
    "build_meta_detection_test_support_set_loader",
    "build_detection_test_loader",
]

logger = logging.getLogger(__name__)


class MetaDatasetFromDict(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    Each time it generates random.
    episodic_test_continualsupportset is only for base class while it is turned on, then the cls code will use all available shots
    """

    def __init__(
        self,
        multi_dataset: Dict,
        # episodic_test_supportset, episodic_test_queryset, episodic_test_continualsupportset
        stage: str = "episodic_train_both",
        num_class: int = 5,
        num_shot: int = 5,
        num_query_shot: int = 1,
        copy: bool = True,
        serialize: bool = False,
        meta_test_seed=0,
    ):
        """
        Args:
            lsts (list): a list which contains a list of elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        # used to refer to all annotation
        self.img2annotation = {
            record["image_id"]: record for record in multi_dataset[-1]
        }
        self.query = multi_dataset[-1]  # deepcopy(multi_dataset[-1])
        # self.test_support_set = None
        # if "test_support_set_anno" in multi_dataset:
        #     self.test_support_set = multi_dataset["test_support_set_anno"][
        #         f"{num_shot}shot"
        #     ]

        self._addrs = []
        self._metadata = multi_dataset["metadata"]
        self._classes = deepcopy(self._metadata["thing_classes"])
        self._thing_dataset_id_to_contiguous_id = deepcopy(
            self._metadata["thing_dataset_id_to_contiguous_id"]
        )
        self._class_ids = list(self._thing_dataset_id_to_contiguous_id.keys())

        # few shot support set dataset
        self.multi_dataset = {
            cid: dataset  # deepcopy(dataset), decrese the memory usage
            for cid, dataset in multi_dataset.items()
            # cid != "metadata" and cid != "test_support_set_anno"
            if cid != -1 and isinstance(cid, int)
        }
        if "test" in stage:
            logger.info("get continual support set")
            self.continual_support_set = multi_dataset[
                "support_set_inference_mode"] if "support_set_inference_mode" in multi_dataset else None
        # for all base classes, one image one annotation
        self._contiguous_id_to_thing_dataset_id = {
            cid: tid for tid, cid in self._thing_dataset_id_to_contiguous_id.items()
        }

        self._copy = copy
        self._serialize = serialize
        self._num_class = num_class
        self._num_shot = num_shot
        self._num_query_shot = num_query_shot
        self._stage = stage
        self._meta_test_seed = meta_test_seed
        self.seed = 2021 + meta_test_seed  # different set for each test

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            def _serialize_one_list(lst: List):
                logger = logging.getLogger(__name__)
                logger.info(
                    "Serializing {} elements to byte tensors and concatenating them all ...".format(
                        len(lst)
                    )
                )
                serialized_lst = [_serialize(x) for x in lst]
                _addr = np.asarray([len(x) for x in lst], dtype=np.int64)
                _addr = np.cumsum(_addr)
                serialized_lst = np.concatenate(serialized_lst)
                logger.info("Serialized dataset takes {:.2f} MiB".format(
                    len(serialized_lst) / 1024 ** 2))

    # def __len__(self):
    #     if self._serialize:
    #         return len(self._addr)
    #     else:
    #         return len(self._lst)

    # def __getitem__(self, idx):
    #     if self._serialize:
    #         start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
    #         end_addr = self._addr[idx].item()
    #         bytes = memoryview(self._lst[start_addr:end_addr])
    #         return pickle.loads(bytes)

        # TODO: serialize
        if self._serialize:
            raise NotImplementedError("_serialize is not implemented.")

    def _get_episodic_train_len(self):
        # TODO: set the length to the maximum iterations to ensure each time the training is different
        if self._serialize:
            raise NotImplementedError("_serialize is not implemented.")
        else:
            return len(self.multi_dataset)

    def _get_episodic_test_support_set_len(self):
        return len(self.multi_dataset)

    def _get_episodic_test_continual_support_set_len(self):
        return len(self.continual_support_set)

    def _get_episodic_test_query_set_len(self):
        return len(self.query)

    # length should be the number of iterations we want to run
    def __len__(self):
        if self._stage == "episodic_train_both":
            return self._get_episodic_train_len()
        elif self._stage == "episodic_test_supportset":
            return self._get_episodic_test_support_set_len()
        elif self._stage == "episodic_test_queryset":
            return self._get_episodic_test_query_set_len()
        elif self._stage == "episodic_test_continualsupportset":
            return self._get_episodic_test_continual_support_set_len()
        else:
            raise NotImplementedError(f"{self._stage} is not supported")

    def _construct_episodic_train_item(self, idx):
        if self._serialize:
            raise NotImplementedError("NotImplemented for serialize")
        else:
            assert idx < len(
                self._classes), f"idx: {idx} is not within {len(self._classes)}"
            class_id = idx  # contiguous class id, from [0, c-1]
            if (
                len(self.multi_dataset[class_id])
                >= self._num_shot + self._num_query_shot
            ):
                # sampling w/o replacement
                support_query_set = np.random.choice(
                    self.multi_dataset[class_id],
                    self._num_shot + self._num_query_shot,
                    replace=False,
                )
            else:
                # sampling with replacement to have fixed length for training
                support_query_set = np.random.choice(
                    self.multi_dataset[class_id],
                    self._num_shot + self._num_query_shot,
                    replace=True,
                )
            support_set = support_query_set[: self._num_shot]
            query_set = []
            for query in support_query_set[self._num_shot:]:
                image_id = query["image_id"]
                item = copy.deepcopy(self.img2annotation[image_id])
                query_set.append(item)

            if self._copy:
                return {
                    "support_set": copy.deepcopy(support_set),
                    "query_set": copy.deepcopy(query_set),
                    "support_set_target": class_id,  # use contigous id
                }
            else:
                return {
                    "support_set": support_set,
                    "query_set": query_set,
                    "support_set_target": class_id,
                }

    def _construct_episodic_test_support_set_item_continual(self, idx):
        # idx does not represent the class id anymore
        item = self.continual_support_set[idx]
        class_continous_id = item["support_set_target"]
        item["class_name"] = self._classes[class_continous_id]
        return item

    def _construct_episodic_test_support_set_item(self, idx):
        # idx will not be used
        assert len(self.multi_dataset) > idx

        if self._serialize:
            raise NotImplementedError("NotImplemented for serialize")
        else:
            class_id = idx  # contiguous class id, from [0, c-1]
            # dataset_id = self._contiguous_id_to_thing_dataset_id[idx]
            assert idx < len(
                self._classes), f"idx: {idx} is not within {len(self._classes)}"
            class_name = self._classes[idx]
            # if self.test_support_set:
            #     assert isinstance(self._meta_test_seed, int), "meta_test_seed is not set"
            #     support_set = self.test_support_set[f"seed{self._meta_test_seed}"][
            #         class_name
            #     ]
            # else:

            # set RNG seed to have deterministic random choices for testing
            with temp_seed(self.seed + idx):
                if len(self.multi_dataset[class_id]) >= self._num_shot:
                    support_set = np.random.choice(
                        self.multi_dataset[class_id], self._num_shot, replace=False
                    )
                else:
                    # TODO: change the process to adapt to a flexible number of samples
                    # support_set = np.random.choice(self.multi_dataset[class_id], len(self.multi_dataset[class_id]), replace=False)
                    support_set = np.random.choice(
                        self.multi_dataset[class_id], self._num_shot, replace=True
                    )
                # chosen_ids = [record["image_id"] for record in support_set]
            if self._copy:
                return {
                    "support_set": copy.deepcopy(support_set),
                    "support_set_target": copy.deepcopy(class_id),
                    "class_name": class_name,
                }
            else:
                return {
                    "support_set": support_set,
                    "support_set_target": class_id,
                    "class_name": class_name,
                }

    def __getitem__(self, idx):
        if self._stage == "episodic_train_both":
            return self._construct_episodic_train_item(idx)
        elif self._stage == "episodic_test_supportset":
            return self._construct_episodic_test_support_set_item(idx)
        elif self._stage == "episodic_test_queryset":
            return self.query[idx]
        elif self._stage == "episodic_test_continualsupportset":
            return self._construct_episodic_test_support_set_item_continual(idx)
            # raise NotImplementedError(f"{self._stage} is not supported")
        else:
            raise NotImplementedError(f"{self._stage} is not supported")


def get_meta_detection_dataset_dicts(
    names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    Modified from from detectron2.data.build import get_detection_dataset_dicts

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts, key is class, value is a dataset dict, following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names) <= 1, names  # currently only support maximum one dataset
    dataset_dicts = DatasetCatalog.get(names[0])
    assert len(dataset_dicts), "Dataset '{}' is empty!".format(names[0])

    # Check
    try:
        check_metadata_consistency("thing_classes", names)
    except AttributeError:  # class names are not available for this dataset
        pass

    assert proposal_files is None, "proposal files are not None"

    metadata = dataset_dicts["metadata"]
    num_classes = len(metadata["thing_classes"])
    for class_i in range(-1, num_classes):
        has_instances = "annotations" in dataset_dicts[class_i][0]
        if filter_empty and has_instances:
            dataset_dicts[class_i] = filter_images_with_only_crowd_annotations(
                dataset_dicts[class_i]
            )
        assert len(
            dataset_dicts[class_i]
        ), f"No valid data found in cls {class_i} with dataset names {names}."
    return dataset_dicts


class MetaMapDataset(MapDataset):
    # disable fallback
    def __getitem__(self, idx):
        """
        idx represents a class id instead of a particular image in this MapDataset
        """
        retry_count = 0
        cur_idx = int(idx)
        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                return data

            # _map_func fails for this idx, try it again
            retry_count += 1
            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warn(f"data: {self._dataset[cur_idx]}, idx: {cur_idx}")
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )
                return None


def _meta_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_meta_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        num_classes = len(
            [k for k in dataset.keys() if k != -1 and isinstance(k, int)])
        logger.info(f"training sampler: {num_classes}")
        if sampler_name == "TrainingSampler":
            # -1 is the image to all annotation, and metadata
            # remaining len is the num of categories
            sampler = TrainingSampler(num_classes)
        elif sampler_name == "SupportSetRepeatFactorTrainingSampler":
            from sylph.data.dataset_sampler.sampler import (
                SupportSetRepeatFactorTrainingSampler,
            )

            repeat_factors = SupportSetRepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD, num_classes
            )
            sampler = SupportSetRepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError(
                "Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,  # a dict
        "sampler": sampler,
        "mapper": mapper,
        # used for query image loader, one for each
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_class": cfg.MODEL.META_LEARN.CLASS,
        "num_shot": cfg.MODEL.META_LEARN.SHOT,  # USE train_shot
        "num_query_shot": cfg.MODEL.META_LEARN.QUERY_SHOT,
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_meta_train_loader_from_config)
def build_meta_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    num_class=5,
    num_shot=5,
    num_query_shot=1,
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    assert isinstance(
        dataset, dict
    ), f"dataset is not dict, {type(dataset)}, first example, {dataset[0]}"
    dataset = MetaDatasetFromDict(
        dataset,
        stage="episodic_train_both",
        num_class=num_class,  # TODO: delete num_class as it is not used here
        num_shot=num_shot,
        num_query_shot=num_query_shot,
        copy=False,
    )

    logger.info(f"meta detection train dataset, len: {len(dataset)}")

    if mapper is not None:
        logger.info(f"mapper is not None: {type(mapper).__name__}")
        dataset = MetaMapDataset(dataset, mapper)

    if sampler is None:
        sampler = TrainingSampler(len(dataset))

    # yiled to a list of length total_batch_size, each data is a dict with "support_set" and "query_set"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def _test_support_set_loader_from_config(cfg, dataset_name: str, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_meta_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
    )

    if mapper is None:
        # mapper is already set up
        mapper = DatasetMapper(cfg, False)

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "num_class": cfg.MODEL.META_LEARN.CLASS,
        "num_shot": cfg.MODEL.META_LEARN.EVAL_SHOT,  # replace it to eval_shot
        "num_query_shot": cfg.MODEL.META_LEARN.QUERY_SHOT,
    }


@configurable(from_config=_test_support_set_loader_from_config)
def build_meta_detection_test_support_set_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    num_workers=0,
    num_class=5,
    num_shot=5,
    num_query_shot=1,
    meta_test_seed=0,
):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    assert isinstance(
        dataset, dict
    ), f"dataset is not dict, {type(dataset)}, first example, {dataset[0]}"
    dataset = MetaDatasetFromDict(
        dataset,
        stage="episodic_test_supportset",
        num_shot=num_shot,
        copy=False,
        meta_test_seed=meta_test_seed,
    )

    logger.info(
        f"meta detection test support set dataset, len: {len(dataset)}")

    if mapper is not None:
        dataset = MetaMapDataset(dataset, mapper)
    if sampler is None:
        logger.info(f"get support set sampler: {len(dataset)}")
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)
    # num_workers = 0

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def _test_support_set_base_loader_from_config(cfg, dataset_name: str, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_meta_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
    )

    if mapper is None:
        # mapper is already set up
        mapper = DatasetMapper(cfg, False)

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        # "num_class": cfg.MODEL.META_LEARN.CLASS,
        # "num_shot": cfg.MODEL.META_LEARN.EVAL_SHOT,  # replace it to eval_shot
        # "num_query_shot": cfg.MODEL.META_LEARN.QUERY_SHOT,
    }


@configurable(from_config=_test_support_set_base_loader_from_config)
def build_meta_detection_test_support_set_base_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    num_workers=0,
):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    assert isinstance(
        dataset, dict
    ), f"dataset is not dict, {type(dataset)}, first example, {dataset[0]}"
    dataset = MetaDatasetFromDict(
        dataset,
        stage="episodic_test_continualsupportset",
        copy=False,
    )

    logger.info(
        f"meta detection test support set dataset, len: {len(dataset)}")

    if mapper is not None:
        dataset = MetaMapDataset(dataset, mapper)
    if sampler is None:
        logger.info(f"get support set base sampler: {len(dataset)}")
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)
    # num_workers = 0

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_meta_detection_dataset_dicts(
        [dataset_name], filter_empty=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (dict): a dict, and each value is a dataset,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, Dict):
        dataset = MetaDatasetFromDict(
            dataset, stage="episodic_test_queryset", copy=False
        )

    logger.info(f"meta detection test query set dataset, len {len(dataset)}")

    if mapper is not None:
        dataset = MetaMapDataset(dataset, mapper)
    if sampler is None:
        logger.info(f"set test set sampler: {len(dataset)}")
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)

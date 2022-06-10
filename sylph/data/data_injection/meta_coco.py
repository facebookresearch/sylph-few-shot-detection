#!/usr/bin/env python3

import copy
import contextlib
import io
import itertools
import logging
import os
from collections import defaultdict

from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
import json
from detectron2.config import global_cfg
import numpy as np


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_learn_coco"]

# JSON_ANNOTATIONS_DIR = (
#     "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"
# )
from .dataset_path_config import COCO_JSON_ANNOTATIONS_DIR, COCO_IMAGE_ROOT_DIR
COCO_FEW_SHOT_JSON_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/coco_meta_learn/"
logger = logging.getLogger(__name__)


def _read_json_file_few_shot(json_file, cls):
    json_file = PathManager.get_local_path(json_file, force=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    _imgs = coco_api.imgs
    anns = list(coco_api.anns.values())
    imgs = [_imgs[ann["image_id"]] for ann in anns]
    return list(zip(imgs, anns))


def _read_json_file(json_file):
    json_file = PathManager.get_local_path(json_file, force=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
    return imgs_anns


def _gen_dataset_dicts(imgs_anns, image_root, ann_keys, id_map, use_cid=True):
    """
    Get dataset_dicts for query set or for normal batch loading.
    With annotation filtered on id_map
    """
    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert (
                anno["image_id"] == image_id
            ), "annotation and image id does not match"
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] in id_map:
                if use_cid:
                    obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def _gen_dataset_dicts_support_set(imgs_anns, image_root, cid, did):
    """
    Given a contingous class id and category dataset id, get a dataset_dicts
    containing annotation of only one category.

    Used to sample support set images and query set images id(since we know it is
    positive for this class). Because query images can have annotations of other classes
    within an episode, we later replace its annotation with a full-on list and filter
    at batched input on the fly
    """
    ann_keys = ["iscrowd", "bbox", "category_id"]

    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert (
                anno["image_id"] == image_id
            ), "annotation and image id does not match"
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] == did:
                obj["category_id"] = cid
                objs.append(obj)
        record["annotations"] = objs
        # Ensure for each class, we use images with at least one annotation
        if len(objs) > 0:
            dataset_dicts.append(record)
    return dataset_dicts


def _gen_dataset_dicts_support_set_filter(imgs_anns, image_root, id_map):
    """
    Filter annotations by id
    """
    ann_keys = ["iscrowd", "bbox", "category_id"]
    support_set_dict = defaultdict(list)
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = defaultdict(list)  # cid: list of objects
        for anno in anno_dict_list:
            assert (
                anno["image_id"] == image_id
            ), "annotation and image id does not match"
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] in id_map:
                obj["category_id"] = id_map[obj["category_id"]]
                objs[obj["category_id"]].append(obj)
        # pass the multiple objs to different list
        for cid, obj_lst in objs.items():
            # update the annotation to the filtered list in record
            support_set_dict[cid].append(
                {**record, **({"annotations": obj_lst})})
    return support_set_dict


def _gen_dataset_dicts_ann_by_category(imgs_anns, image_root, id_map, sample_size: int):
    """
    Filter annotations by id. Return image_id indexed records
    """
    ann_keys = ["iscrowd", "bbox", "category_id"]
    support_set_dict = defaultdict(list)  # category id: anno_lst
    images = defaultdict(dict)  # image_id: record dict
    for (img_dict, anno_dict_list) in imgs_anns:
        image_id = img_dict["id"]
        if image_id not in images:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]
            images[image_id] = record

        for anno in anno_dict_list:
            assert (
                anno["image_id"] == image_id
            ), "annotation and image id does not match"
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            cid = obj["category_id"]
            obj["image_id"] = image_id
            if cid in id_map:
                support_set_dict[cid].append(obj)
    record_dict = defaultdict(dict)  # image id to anno

    # sample annotations, and then put annotations back to link by images
    for _, ann_lst in support_set_dict.items():
        downsampled_ann_lst = np.random.choice(
            ann_lst, min(len(ann_lst), sample_size), replace=False)
        # downsampled_ann_lst = ann_lst[0 : min(len(ann_lst), sample_size)]
        for ann in downsampled_ann_lst:
            image_id = ann["image_id"]
            if image_id not in record_dict:
                record = images[image_id]  # image record
                record["annotations"] = [ann]
                record_dict[image_id] = record
            else:
                record_dict[image_id]["annotations"].append(ann)
    return record_dict


def load_pretrain_coco_json(json_file, json_root, image_root, metadata, dataset_name):
    logger.info(f"load_pretrain_coco_json: {json_file}")

    name, meta_training_stage, training_stage, split = dataset_name.split("_")
    imgs_anns = _read_json_file(json_file)
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    ann_keys = ["iscrowd", "bbox", "category_id"]
    # convert each annotation to record, filter the annotation by id_map
    logger.info(f"Pretraining {training_stage} stage")
    if training_stage == "train":
        if split == "base" or split == "novel":
            dataset_dicts = _gen_dataset_dicts(
                imgs_anns, image_root, ann_keys, id_map)
            logger.info(f"training on {split} split.")

        elif split == "all":
            print(
                f"joint training on all classes, where global_cfg.MODEL.TFA.TRAIN_SHOT is set to {global_cfg.MODEL.TFA.TRAIN_SHOT}")
            base_id_map = metadata["base_thing_dataset_id_to_contiguous_id"]
            novel_id_map = metadata["novel_thing_dataset_id_to_contiguous_id"]
            # id is still object id
            base_dataset_dicts = _gen_dataset_dicts(
                imgs_anns, image_root, ann_keys, base_id_map, use_cid=False
            )
            # For novel classes, sample 10 shots only
            record_dict = _gen_dataset_dicts_ann_by_category(
                imgs_anns, image_root, novel_id_map, sample_size=global_cfg.MODEL.TFA.TRAIN_SHOT
            )
            base_datasets = defaultdict(dict)  # image id: record list
            for item in base_dataset_dicts:
                if item["image_id"] in base_datasets:
                    raise ValueError("should not repeat")
                base_datasets[item["image_id"]] = item
            # merge record list
            for img_id, record in record_dict.items():
                if img_id in base_datasets:
                    base_datasets[img_id]["annotations"] += record["annotations"]
                else:
                    base_datasets[img_id] = record
            # replace all cid
            for _, record in base_datasets.items():
                anno_list = record["annotations"]
                for i, ann in enumerate(anno_list):
                    cid = ann["category_id"]
                    anno_list[i]["category_id"] = id_map[cid]

            dataset_dicts = list(itertools.chain(base_datasets.values()))
        else:
            raise NotImplementedError(f"{split} is not supported")
    elif training_stage == "finetune":  # a finetune stage on all categories
        # assert split == "all", "finetune stage, but the split is not 'all'"
        print("TFA finetune")
        record_dict = _gen_dataset_dicts_ann_by_category(
            imgs_anns, image_root, id_map, sample_size=global_cfg.MODEL.TFA.TRAIN_SHOT
        )
        # replace all cid
        for _, record in record_dict.items():
            anno_list = record["annotations"]
            for i, ann in enumerate(anno_list):
                cid = ann["category_id"]
                anno_list[i]["category_id"] = id_map[cid]
        dataset_dicts = list(itertools.chain(record_dict.values()))
    else:  # "val"  for validation, always use all annotations
        dataset_dicts = _gen_dataset_dicts(
            imgs_anns, image_root, ann_keys, id_map)
    import os
    if os.environ.get('SYLPH_TEST_MODE', default="False"):
        logger.warn(
            "SYLPH_TEST_MODE on, only load 10 images in pretraining stage, both train and val will be impacted")
        return copy.deepcopy(dataset_dicts[0:10])
    return dataset_dicts


def load_few_shot_coco_json(json_file, json_root, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
                         Only used for pretraining
                         Fixated in the function for support set and query set
        json_root (str): Only used for pretraining
        image_root (str): the directory where the images in this json file exists.
                          Only set for support set, for query set, its hard written here
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    name, meta_training_stage, training_stage, split = dataset_name.split("_")
    meta_learn = True if meta_training_stage == "meta" else False
    if not meta_learn:
        return load_pretrain_coco_json(
            json_file, json_root, image_root, metadata, dataset_name
        )

    dataset_dicts = {}
    dataset_dicts["metadata"] = copy.deepcopy(metadata)

    # Step 1: Load json file for meta-learning
    # Support set are always processed from instances_train
    support_set_file = os.path.join(
        COCO_JSON_ANNOTATIONS_DIR, "instances_train2017.json"
    )
    logger.info(f"{dataset_name}, support set file: {support_set_file}")
    support_set_annotation = _read_json_file(support_set_file)

    # Step 2. prepare annotations/images for query set
    # In train: query set are from instances_train
    # In test: query set are from instances_val
    query_json_file = os.path.join(
        COCO_JSON_ANNOTATIONS_DIR, f"instances_{training_stage}2017.json"
    )
    query_image_root = os.path.join(
        COCO_IMAGE_ROOT_DIR, f"{training_stage}2017")
    # in meta-training/finetuning stage: use this for query set reference
    logger.info(f"{dataset_name}, query annotation file: {query_json_file}")
    query_set_annotation = _read_json_file(
        query_json_file
    )  # used for traing or testing

    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    ann_keys = ["iscrowd", "bbox", "category_id"]

    # Step 3: prepare list of data items from annotation, need image_root
    dataset_dicts.update(
        _gen_dataset_dicts_support_set_filter(
            support_set_annotation, image_root, id_map
        )
    )
    if split == "all":  # downsample
        novel_id_map = metadata["novel_thing_dataset_id_to_contiguous_id"]
        for ndid in novel_id_map.keys():
            cid = id_map[ndid]
            dataset_dicts[cid] = np.random.choice(
                dataset_dicts[cid], global_cfg.MODEL.META_LEARN.EVAL_SHOT, replace=False)

    # 2. get query dataset
    dataset_dicts[-1] = _gen_dataset_dicts(
        query_set_annotation, query_image_root, ann_keys, id_map
    )
    # If it is validation, we save json file that only has novel or base classes for evaluator
    # # TODO: check if this code is needed
    # # From here is to support Xinyu's repeat tests
    # predefined_test_support_set_mapper = {
    #     "coco_meta_val_base": os.path.join(
    #         COCO_FEW_SHOT_JSON_ANNOTATIONS_DIR,
    #         "coco/coco_meta_val_base_support_set.json",
    #     ),
    #     "coco_meta_val_novel": os.path.join(
    #         COCO_FEW_SHOT_JSON_ANNOTATIONS_DIR,
    #         "coco/coco_meta_val_novel_support_set.json",
    #     ),
    # }
    # if dataset_name in predefined_test_support_set_mapper:
    #     test_support_set_anno_path = predefined_test_support_set_mapper[dataset_name]
    #     assert PathManager.exists(
    #         test_support_set_anno_path
    #     ), f"File {test_support_set_anno_path} does not exist."
    #     logger.info(f"load test support set {test_support_set_anno_path}")
    #     test_support_set_anno_path = PathManager.get_local_path(
    #         test_support_set_anno_path, force=True
    #     )
    #     with open(test_support_set_anno_path, "r") as f:
    #         test_support_set_anno = json.load(f)  # Dict
    #         # test support set anno's structure:
    #         # [${x}shot][seed${y}][${class_name}]
    #     dataset_dicts["test_support_set_anno"] = test_support_set_anno
    return dataset_dicts


def register_meta_learn_coco(name, metadata, imgdir, jsondir, annofile):
    split = name.split("_")[-1]  # base/novel/all
    print(f"coco split: {split}")
    metadata["thing_dataset_id_to_contiguous_id"] = metadata[
        f"{split}_thing_dataset_id_to_contiguous_id"
    ]
    metadata["thing_classes"] = metadata[f"{split}_thing_classes"]
    metadata["thing_colors"] = metadata[f"{split}_thing_colors"]
    # make sure to deepcopy metadata as it will change with the datasetname
    DatasetCatalog.register(
        name,
        lambda: load_few_shot_coco_json(
            annofile, jsondir, imgdir, copy.deepcopy(metadata), name
        ),
    )
    return copy.deepcopy(metadata)

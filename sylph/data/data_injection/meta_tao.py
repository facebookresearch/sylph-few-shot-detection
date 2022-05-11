# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
from typing import List

from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)
JSON_ANNOTATIONS_DIR = (
    "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"
)
from sylph.data.data_injection.meta_lvis import (
    _load_lvis_json,
    _gen_dataset_dicts_support_set_filter,
)


def get_tao_file_name(img_root, img_dict):
    if "file_name" in img_dict:
        file_path = os.path.join(img_root, img_dict["file_name"])
        # if not PathManager.exists(file_path):
        #     raise ValueError(file_path)
        return file_path
    else:
        raise ValueError(f"{img_dict}")


def _gen_dataset_dicts(imgs_anns, image_root, id_map, ann_keys: List[str]):
    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_tao_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            if anno["category_id"] not in id_map:
                continue
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {key: anno[key] for key in ann_keys}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            obj["category_id"] = id_map[anno["category_id"]]
            segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            valid_segm = [
                poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
            ]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def load_tao_json(json_file, image_root, meta, anno_keys, dataset_name=None):
    imgs_anns = _load_lvis_json(json_file)
    return _gen_dataset_dicts(
        imgs_anns, image_root, meta["thing_dataset_id_to_contiguous_id"], anno_keys
    )


LVIS_FEW_SHOT_JSON_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/lvis_meta_learn/"


def load_few_shot_tao_json(query_json_file, image_root, metadata, dataset_name):
    ann_keys = ["iscrowd", "bbox", "category_id"]

    name, meta_training_stage, training_stage, split = dataset_name.split("_")
    # assert meta_training_stage == "meta"
    assert training_stage == "val"
    if meta_training_stage == "pretrain":
        return load_tao_json(query_json_file, image_root, metadata, ann_keys, dataset_name)

    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    dataset_dicts = {}
    dataset_dicts["metadata"] = copy.deepcopy(metadata)
    # Step 1: Use lvis train set for support set file
    support_set_file = os.path.join(JSON_ANNOTATIONS_DIR, "lvis/lvis_v1_train.json")
    support_set_image_root = "memcache_manifold://fair_vision_data/tree/coco_"
    logger.info(f"{dataset_name}, support set file: {support_set_file}")
    support_set_annotation = _load_lvis_json(support_set_file)

    # Step 2. prepare annotations for query set
    logger.info(f"{dataset_name}, query annotation file: {query_json_file}")
    query_set_annotation = _load_lvis_json(query_json_file)

    # Step 3: prepare list of data items from annotation
    dataset_dicts.update(
        _gen_dataset_dicts_support_set_filter(
            support_set_annotation,
            support_set_image_root,
            id_map,
            dataset_name,
        )
    )

    dataset_dicts[-1] = _gen_dataset_dicts(
        query_set_annotation, image_root, id_map, ann_keys
    )
    logger.info(f"example: {dataset_dicts[-1][0]}")
    del support_set_annotation
    del query_set_annotation
    return dataset_dicts


def register_meta_learn_tao(name, metadata, imgdir, annofile):
    split = name.split("_")[-1]  # val
    print(f"tao split: {split}")
    metadata["thing_dataset_id_to_contiguous_id"] = metadata[
        f"{split}_dataset_id_to_contiguous_id"
    ]
    metadata["thing_classes"] = metadata[f"thing_{split}_classes"]
    DatasetCatalog.register(
        name,
        lambda: load_few_shot_tao_json(
            query_json_file=annofile,
            image_root=imgdir,  # query image root
            metadata=copy.deepcopy(metadata),
            dataset_name=name,
        ),
    )
    return copy.deepcopy(metadata)

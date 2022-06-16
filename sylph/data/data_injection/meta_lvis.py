"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any

from detectron2.config import global_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

from .classes import unknown_category
import numpy as np
from sylph.data.utils import temp_seed

from .dataset_path_config import LVIS_JSON_ANNOTATIONS_DIR

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)


def _load_lvis_json(json_file):
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds())
        )

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(
            len(imgs_anns), json_file)
    )
    return imgs_anns


def get_file_name(img_root, img_dict):
    # Determine the path including the split folder ("train2017", "val2017", "test2017") from
    # the coco_url field. Example:
    #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
    split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
    return os.path.join(img_root + split_folder, file_name)


def _gen_dataset_dicts(imgs_anns, image_root: str, id_map: Dict[int, Any]):
    """
    Split the annotations into two groups: base classes and novel classes.
    Replace novel classes with unknown category id
    """
    dataset_dicts = []
    use_unknown = unknown_category["id"] in id_map
    logger.info(f"use_unknown: {use_unknown}")
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        all_cats = set()
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            # handle category_id
            if anno["category_id"] in id_map:  # base classes
                obj["category_id"] = id_map[anno["category_id"]]
                all_cats.add(anno["category_id"])
            else:  # novel classes, mark it as unknown
                if use_unknown:
                    obj["category_id"] = id_map[unknown_category["id"]]
                else:
                    continue
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
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
        record["annotations_cat_set"] = all_cats
        dataset_dicts.append(record)
    return dataset_dicts


def _gen_dataset_dicts_ann_by_category(imgs_anns, image_root: str, id_map: Dict[int, Any], sample_size: int):
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
            record["file_name"] = record["file_name"] = get_file_name(
                image_root, img_dict)  # change
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]
            record["not_exhaustive_category_ids"] = img_dict.get(
                "not_exhaustive_category_ids", []
            )
            record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
            images[image_id] = record

        for anno in anno_dict_list:
            assert (
                anno["image_id"] == image_id
            ), "annotation and image id does not match"
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = id_map[anno["category_id"]]
            cid = obj["category_id"]
            obj["image_id"] = image_id
            if cid in id_map:
                support_set_dict[cid].append(obj)
    record_dict = defaultdict(dict)

    for _, ann_lst in support_set_dict.items():  # sample annotations
        downsampled_ann_lst = np.random.choice(
            ann_lst, min(len(ann_lst), sample_size), replace=False)
        # downsampled_ann_lst = ann_lst[0 : min(len(ann_lst), sample_size)]
        # put annotations back to records
        for ann in downsampled_ann_lst:
            image_id = ann["image_id"]
            if image_id not in record_dict:
                record = images[image_id]  # image record
                record["annotations"] = [ann]
                record_dict[image_id] = record
            else:
                record_dict[image_id]["annotations"].append(ann)
    return list(record_dict.values())


def load_lvis_json_many_shots(json_file, image_root, meta, dataset_name=None):
    imgs_anns = _load_lvis_json(json_file)
    return _gen_dataset_dicts(
        imgs_anns, image_root, meta["thing_dataset_id_to_contiguous_id"]
    )

# Added for TFA, finetune the classifier or regressor on all classes, with maximum shots: sample_size


def load_lvis_json_sample_k_per_cat(json_file, image_root, meta, sample_size: int, dataset_name: str = None):
    imgs_anns = _load_lvis_json(json_file)
    return _gen_dataset_dicts_ann_by_category(imgs_anns, image_root, meta["thing_dataset_id_to_contiguous_id"], sample_size=sample_size)


def _gen_dataset_dicts_support_set_filter(imgs_anns, image_root: str, all_id_map: Dict[int, Any], id_map: Dict[int, Any], dataset_name: str, base_eval_shot: int, base_id_map: Dict[int, Any] = None):
    """
    For evaluation, each only sample at most 100 images to decrease memory consumption and avoid oom error
    """
    support_set_dict = defaultdict(list)
    dataset, learning_stage, training_stage, data_split = dataset_name.split(
        '_')
    assert learning_stage == "meta"
    # sort all annotations by category id
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = defaultdict(list)  # save annotations by category
        for anno in anno_dict_list:
            if anno["category_id"] not in id_map:
                continue
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            obj["category_id"] = id_map[anno["category_id"]]
            # not supporting segmentation for now
            # segm = anno["segmentation"]  # list[list[float]]
            # # filter out invalid polygons (< 3 points)
            # valid_segm = [
            #     poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
            # ]
            # assert len(segm) == len(
            #     valid_segm
            # ), "Annotation contains an invalid polygon with < 3 points"
            # assert len(segm) > 0
            # obj["segmentation"] = segm
            objs[obj["category_id"]].append(obj)
        # pass the multiple objs to different list
        for cid, obj_lst in objs.items():
            for obj in obj_lst:  # Ensure each record has only one annotation
                support_set_dict[cid].append(
                    {**record, **({"annotations": [obj]})})

    assert len(id_map) == len(
        support_set_dict
    ), f"{len(id_map)} != {len(support_set_dict)}"

    # random sample but deterministic each time
    with temp_seed(2021):
        for cid in support_set_dict.keys():
            np.random.shuffle(support_set_dict[cid])

    # Ensure support_set_dict will not be changed
    # TODO:The code to support generating class codes using all available images in a category, not well tested yet
    datasets_dict = []
    if base_id_map is not None and training_stage == "val" and data_split == "all":
        logger.info(
            "Get the format for generating code using all boxes in a category")
        for cat_id in base_id_map.keys():
            cid = id_map[cat_id]
            record_lst = deepcopy(support_set_dict[cid])
            if base_eval_shot > -1:
                sample_shot = min(len(record_lst), base_eval_shot)
                # random choosing to avoid chosing all annotations from the same images
                record_lst = np.random.choice(
                    record_lst,
                    sample_shot,
                    replace=False,
                )
                # record_lst = record_lst[0: sample_shot]
            total_annos = len(record_lst)
            # separate the list into 10 each
            for i in range(0, total_annos, 10):
                end = min(i+10, total_annos)
                datasets_dict.append(
                    {"support_set": record_lst[i: end], "len": end - i, "total_len": total_annos, "support_set_target": cid})
            # delete from the support set which will be used as novel classes
            # TODO: save only novel categories
            # del support_set_dict[cid]
    # the first is for training, we can do random sampling, and the second is for inference
    return support_set_dict, datasets_dict

    # return support_set_dict

# def _gen_dataset_dicts_support_set_filter(imgs_anns, image_root: str, id_map:Dict[int, Any], dataset_name: str):
#     """
#     For evaluation, each only sample at most 100 images to decrease memory consumption and avoid oom error
#     in fblearner flow.
#     """
#     support_set_dict = defaultdict(list)
#     limit_size = False # if "meta_val" in dataset_name else False
#     sample_size = 30
#     for (img_dict, anno_dict_list) in imgs_anns:
#         record = {}
#         record["file_name"] = get_file_name(image_root, img_dict)
#         record["height"] = img_dict["height"]
#         record["width"] = img_dict["width"]
#         record["not_exhaustive_category_ids"] = img_dict.get(
#             "not_exhaustive_category_ids", []
#         )
#         record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
#         image_id = record["image_id"] = img_dict["id"]

#         objs = defaultdict(list)  # save annotation by category
#         for anno in anno_dict_list:
#             if anno["category_id"] not in id_map:
#                 continue
#             # Check that the image_id in this annotation is the same as
#             # the image_id we're looking at.
#             # This fails only when the data parsing logic or the annotation file is buggy.
#             assert anno["image_id"] == image_id
#             obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
#             # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
#             # variable will have a field with COCO-specific category mapping.
#             obj["category_id"] = id_map[anno["category_id"]]
#             segm = anno["segmentation"]  # list[list[float]]
#             # filter out invalid polygons (< 3 points)
#             valid_segm = [
#                 poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
#             ]
#             assert len(segm) == len(
#                 valid_segm
#             ), "Annotation contains an invalid polygon with < 3 points"
#             assert len(segm) > 0
#             obj["segmentation"] = segm
#             objs[obj["category_id"]].append(obj)
#         # pass the multiple objs to different list
#         for cid, obj_lst in objs.items():
#             support_set_dict[cid].append({**record, **({"annotations": obj_lst})})
#     if limit_size:  # random downs sampling
#         import numpy as np
#         from sylph.data.utils import temp_seed
#         for cid, cid_anno_lst in support_set_dict.items():
#             with temp_seed(2021 + cid):
#                 support_set_dict[cid] = np.random.choice(
#                     cid_anno_lst, min(len(cid_anno_lst), sample_size), replace=False
#                 )
#     assert len(id_map) == len(
#         support_set_dict
#     ), f"{len(id_map)} != {len(support_set_dict)}"
#     return support_set_dict


def load_few_shot_lvis_json(json_file, image_root, metadata, dataset_name):
    name, meta_training_stage, training_stage, split = dataset_name.split("_")
    meta_learn = True if meta_training_stage == "meta" else False
    if not meta_learn:
        if training_stage == "finetune":  # this will use sample_size on base classes + all novelr
            print("finetune stage")
            assert split == "all", f"training split is not 'all', but: {split}"
            return load_lvis_json_sample_k_per_cat(json_file, image_root, metadata, sample_size=global_cfg.MODEL.TFA.TRAIN_SHOT, dataset_name=dataset_name)
        else:
            return load_lvis_json_many_shots(json_file, image_root, metadata, dataset_name)

    # Meta-learn
    # current evaluation class ids
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    # both base and novel class ids
    all_id_map = metadata["all_dataset_id_to_contiguous_id"]
    # get base classes
    base_id_map = None
    if meta_learn and training_stage == "val" and split == "all" and global_cfg.MODEL.META_LEARN.USE_ALL_GTS_IN_BASE_CLASSES:  # get base classes
        base_data_split = global_cfg.DATASETS.TRAIN[0].split("_")[-1]
        base_id_map = metadata[f"{base_data_split}_dataset_id_to_contiguous_id"]
    dataset_dicts = {}
    dataset_dicts["metadata"] = copy.deepcopy(metadata)
    # Step 1: Load json file for meta-learning
    # Support set are always processed from instances_train
    support_set_file = os.path.join(
        LVIS_JSON_ANNOTATIONS_DIR, "lvis_v1_train.json")

    logger.info(f"{dataset_name}, support set file: {support_set_file}")
    support_set_annotation = _load_lvis_json(support_set_file)

    # Step 2. prepare annotations for query set
    # In train: query set are from instances_train
    # In test: query set are from instances_val
    json_file = os.path.join(
        LVIS_JSON_ANNOTATIONS_DIR, f"lvis_v1_{training_stage}.json"
    )
    logger.info(f"{dataset_name}, query annotation file: {json_file}")
    query_set_annotation = _load_lvis_json(
        json_file)  # used for traing or testing

    # Step 3: prepare list of data items from annotation, need image_root
    # 1. get support set
    base_eval_shot = global_cfg.MODEL.META_LEARN.BASE_EVAL_SHOT
    logger.info(f"{dataset_name}, base_eval_shot: {base_eval_shot}")

    support_set_cat_to_list_dict, support_set_list_inference = _gen_dataset_dicts_support_set_filter(
        support_set_annotation, image_root, all_id_map, id_map, dataset_name, base_eval_shot, base_id_map
    )
    dataset_dicts.update(
        support_set_cat_to_list_dict
    )
    dataset_dicts["support_set_inference_mode"] = support_set_list_inference
    logger.info(f"{dataset_name}, {dataset_dicts.keys()}")

    # 2. get query dataset
    # TODO: change it to "query_set"
    dataset_dicts[-1] = _gen_dataset_dicts(
        query_set_annotation, image_root, id_map)
    # Downsample the validation size while in testing mode
    if os.environ.get("SYLPH_TEST_MODE", default=False):
        dataset_dicts[-1] = deepcopy(dataset_dicts[-1][0:10])
        logger.info("Downsample the validation size to only 10.")

    del support_set_annotation
    del query_set_annotation
    return dataset_dicts


def register_meta_learn_lvis(name, metadata, imgdir, annofile):
    split = name.split("_")[-1]  # base/common/rare/all, basemix/novelmix
    print(f"lvis split: {split}")
    metadata["thing_dataset_id_to_contiguous_id"] = metadata[
        f"{split}_dataset_id_to_contiguous_id"
    ]
    metadata["thing_classes"] = metadata[f"thing_{split}_classes"]
    # make sure to deepcopy metadata as it will change with the datasetname
    DatasetCatalog.register(
        name,
        lambda: load_few_shot_lvis_json(
            annofile, imgdir, copy.deepcopy(metadata), name
        ),
    )
    return copy.deepcopy(metadata)


if __name__ == "__main__":
    """
    Test the LVIS json dataset loader.

    Usage:
        python -m detectron2.data.datasets.lvis \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    from PIL import Image

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_lvis_json_many_shots(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "lvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts[: int(sys.argv[4])]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)

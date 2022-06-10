#!/usr/bin/env python3

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
Here we only register the few-shot datasets and complete COCO, PascalVOC and
LVIS have been handled by the builtin datasets in detectron2.
"""

import copy
import logging
import os

from detectron2.data import MetadataCatalog
from sylph.data.data_injection.builtin_meta_dataset_few_shot_detection import (
    _fewshot_get_builtin_metadata,
)
from sylph.data.data_injection.meta_coco import register_meta_learn_coco
from sylph.data.data_injection.meta_lvis import register_meta_learn_lvis
from sylph.data.data_injection.meta_tao import register_meta_learn_tao
from .dataset_path_config import COCO_IMAGE_ROOT_DIR, COCO_JSON_ANNOTATIONS_DIR, LVIS_JSON_ANNOTATIONS_DIR
# JSON_ANNOTATIONS_DIR = (
#     "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"
# )

# JSON_ANNOTATIONS_DIR = (
#     "datasets/coco/annotations/"
# )
# IMAGE_ROOT_DIR = "datasets/coco/"

logger = logging.getLogger(__name__)

"""
Meta-learning dataset name convention follows: [datasetname]_[train_stage]_[train/test]_[datasplit]
"""


def register_all_coco_meta_learn(
    # few_shot_root="manifold://fai4ar/tree/datasets/coco_meta_learn",
):
    """
    pretrain: the data loader needs to filter out novel classes
    """
    # register few-shot coco datase
    METASPLITS = [
        (
            "coco_pretrain_train_base",  # pretraining, only 60 base classes is available
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),  # image root
            "instances_train2017.json",  # json file, will add it to annotation_dir later
        ),
        (
            "coco_pretrain_train_novel",  # pretraining, only 60 base classes is available
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),  # image root
            "instances_train2017.json",  # json file, will add it to annotation_dir later
        ),
        (
            "coco_pretrain_finetune_all",  # finetune, used in TFA finetune stage
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),  # image root
            "instances_train2017.json",  # json file, will add it to annotation_dir later
        ),
        (
            # finetune, used in TFA simplified finetune stage, use only novel categories
            "coco_pretrain_finetune_novel",
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),  # image root
            "instances_train2017.json",  # json file, will add it to annotation_dir later
        ),
        (
            "coco_pretrain_val_base",  # pretraining, only 60 base classes is available
            os.path.join(COCO_IMAGE_ROOT_DIR, "val2017"),
            "instances_val2017.json",
        ),
        (
            "coco_pretrain_val_novel",  # pretraining, only 60 base classes is available
            os.path.join(COCO_IMAGE_ROOT_DIR, "val2017"),
            "instances_val2017.json",
        ),
        # Pretain all classes
        (
            "coco_pretrain_train_all",  # novel classes use only 10 shots
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            "instances_train2017.json",
        ),
        (
            "coco_pretrain_val_all",  # validate on base and novel
            os.path.join(COCO_IMAGE_ROOT_DIR, "val2017"),
            "instances_val2017.json",
        ),
        (
            "coco_meta_train_base",  # meta training, trains stage, only 60 base classes
            # image root for both support set and query set
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            None,  # json file, will add in load_few_shot_coco_json
        ),
        # meta-finetune all classes # novel classes only 10 shots
        (
            # meta training, trains stage, all 80 classes, but for novel set, will only sample a few
            "coco_meta_train_all",
            # image root for both support set and query set
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            None,  # json file, will add in load_few_shot_coco_json
        ),
        (
            "coco_meta_val_novel",  # meta training, validation stage, use 20 novel classes
            # image root for support set
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            None,  # cant be none for evaluator, later we dynamically generate
        ),
        (
            "coco_meta_val_base",  # meta training, validation stage, use 60 base classes
            # image root for support set
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            None,  # cant be none for evaluator, later we dynamically generate
        ),
        (
            "coco_meta_val_all",
            # image root for support set
            os.path.join(COCO_IMAGE_ROOT_DIR, "train2017"),
            None,  # cant be none for evaluator, later we dynamically generate
        ),
    ]

    # register small meta datasets for fine-tuning stage
    metadata = _fewshot_get_builtin_metadata("coco_meta_learn")
    # a common meta data
    for name, imgdir, annofile in METASPLITS:
        new_annofile = (
            os.path.join(COCO_JSON_ANNOTATIONS_DIR, annofile)
            if annofile is not None
            else None
        )
        # different name will update the metadata thing_classes
        updated_metadata = register_meta_learn_coco(
            name=name,
            metadata=copy.deepcopy(metadata),
            imgdir=imgdir,
            # jsondir=few_shot_root,
            annofile=new_annofile,
        )

        # Added for evaluator purpose, use query annotation file
        if "meta_val" in name:
            new_annofile = os.path.join(
                COCO_JSON_ANNOTATIONS_DIR, "instances_val2017.json"
            )

        if new_annofile is not None:
            MetadataCatalog.get(name).set(
                json_file=new_annofile,
                evaluator_type="coco_meta_learn",
                **updated_metadata,
            )
        else:
            MetadataCatalog.get(name).set(
                evaluator_type="coco_meta_learn", **updated_metadata
            )


def register_all_lvis_meta_learn(
    few_shot_root="manifold://fai4ar/tree/datasets/lvis_meta_learn",
):
    """
    All available lvis datasets
    Normal batch training/finetuning/validation:
        lvis_pretrain_{train/val/finetune}_{data_split}
    Meta/episodic learning stage training/validation:
        lvis_meta_{train/val}_{data_split}
    """
    from sylph.data.data_injection.classes import datasplit_categories

    lvis_image_root = COCO_IMAGE_ROOT_DIR
    # TODO: if they can share the same images
    # "memcache_manifold://fair_vision_data/tree/coco_"
    METASPLITS = []
    # register pretrain datasets
    for data_split in datasplit_categories.keys():
        # add TFA finetune stage, it samples images
        for training_stage in ["train", "val", "finetune"]:
            dataset_name = f"lvis_pretrain_{training_stage}_{data_split}"
            json_file = f"lvis_v1_{training_stage}.json" if training_stage != "finetune" else "lvis_v1_train.json"
            METASPLITS.append((dataset_name, lvis_image_root, json_file))
            logger.info(f"Registering {dataset_name}")
    # register meta-learn datasets, including base, novel, and all splits
    for data_split in datasplit_categories.keys():
        for training_stage in ["train", "val"]:
            dataset_name = f"lvis_meta_{training_stage}_{data_split}"
            json_file = None
            METASPLITS.append((dataset_name, lvis_image_root, json_file))
            logger.info(f"Registering {dataset_name}")

    # register small meta datasets for fine-tuning stage
    metadata = _fewshot_get_builtin_metadata("lvis_meta_learn")
    # a common meta data
    for name, imgdir, annofile in METASPLITS:
        new_annofile = (
            os.path.join(LVIS_JSON_ANNOTATIONS_DIR, annofile)
            if annofile is not None
            else None
        )
        # different name will update the metadata thing_classes
        updated_metadata = register_meta_learn_lvis(
            name=name,
            metadata=copy.deepcopy(metadata),
            imgdir=imgdir,
            jsondir=few_shot_root,
            annofile=new_annofile,
        )

        # Added for evaluator purpose
        if "meta_val" in name:
            new_annofile = os.path.join(
                LVIS_JSON_ANNOTATIONS_DIR, "lvis_v1_val.json")

        if new_annofile is not None:
            MetadataCatalog.get(name).set(
                json_file=new_annofile,
                evaluator_type="lvis_meta_learn",
                **updated_metadata,
            )
        else:
            MetadataCatalog.get(name).set(
                evaluator_type="lvis_meta_learn", **updated_metadata
            )


# Temporarily for demo
# TAO_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/TAO/20210625_video_demo/"


# def register_all_tao_meta_learn():
#     """
#     pretrain: the data loader needs to filter out novel classes
#     """
#     # register few-shot coco datase
#     METASPLITS = [
#         (
#             "tao_pretrain_val_base",  # 1105 base classes
#             "manifold://fai4ar/tree/datasets/TAO/frames",  # image root
#             "tao_lvis_val_full_frames.json",  # validation.json",
#         ),
#         (
#             "tao_meta_val_novelpartial",  # train part inference
#             "manifold://fai4ar/tree/datasets/TAO/frames",  # image root
#             "tao_lvis_val.json",
#         ),
#         (
#             "tao_meta_val_novel",  # val part inference
#             "manifold://fai4ar/tree/datasets/TAO/frames",  # image root
#             "tao_lvis_val_full_frames.json",  # validation.json",
#         ),
#     ]

#     # register small meta datasets for fine-tuning stage
#     metadata = _fewshot_get_builtin_metadata("tao_meta_learn")
#     # a common meta data
#     for name, imgdir, annofile in METASPLITS:
#         new_annofile = (
#             os.path.join(TAO_ANNOTATIONS_DIR, annofile)
#             if annofile is not None
#             else None
#         )
#         # different name will update the metadata thing_classes
#         updated_metadata = register_meta_learn_tao(
#             name=name,
#             metadata=copy.deepcopy(metadata),
#             imgdir=imgdir,
#             annofile=new_annofile,  # query json file
#         )

#         if new_annofile is not None:
#             MetadataCatalog.get(name).set(
#                 json_file=new_annofile,
#                 evaluator_type="lvis_meta_learn",
#                 **updated_metadata,
#             )
#         else:
#             MetadataCatalog.get(name).set(
#                 evaluator_type="lvis_meta_learn", **updated_metadata
#             )


register_all_coco_meta_learn()
register_all_lvis_meta_learn()
# register_all_tao_meta_learn()

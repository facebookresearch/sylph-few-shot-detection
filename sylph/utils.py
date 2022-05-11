#!/usr/bin/env python3

import argparse
import io
import json
import os
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["figure.figsize"] = [24, 24]
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer

from .data.data_injection.builtin_dataset_few_shot_detection import (
    register_all_coco_meta_learn,
)

def dataset_id_map(ds_id, metadata):
    return metadata.thing_dataset_id_to_contiguous_id[ds_id]

def PathManagerImgSave(img, path, img_color="RGB"):
    # img : Image
    # path: string
    with PathManager.open(path, "wb") as f:  # this 'rb' is necessary!!!
        img.save(f)


def PathManagerImgLoader(path, img_color="RGB"):
    with PathManager.open(path, "rb") as f:  # this 'rb' is necessary!!!
        f = f.read()
        img = Image.open(io.BytesIO(f)).convert(img_color)
    return img


def create_instances(predictions, image_size,metadata, conf):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > conf).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"], metadata) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
        ret.gt_annotation_ids = [predictions[i]["annotation_id"] for i in chosen]
    except KeyError:
        pass
    return ret

# Handles config file
def merge_from_file_updater(config_file):
    def _update(cfg):
        if config_file:
            cfg.merge_from_file(config_file)
        return cfg

    return _update


def merge_from_list_updater(overwrite_opts):
    def updater(cfg):
        cfg.merge_from_list(overwrite_opts or [])
        return cfg

    return updater


def update_cfg(cfg, *updater_list):
    for updater in updater_list:
        cfg = updater(cfg)
    return cfg


def create_cfg(default_cfg, config_file, overwrite_opts):
    return update_cfg(
        default_cfg,
        merge_from_file_updater(config_file),
        merge_from_list_updater(overwrite_opts),
    )

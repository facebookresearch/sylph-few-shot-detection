#!/usr/bin/env python3
import json
import os
from collections import defaultdict

import detectron2.data.datasets  # noqa # add pre-defined metadata
import numpy as np
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer
from PIL import Image

ann_keys = ["iscrowd", "bbox", "category_id"]
visualization = False


def load_train_data():
    annotation_path = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/coco/instances_train2017.json"
    with PathManager.open(annotation_path, "r") as f:
        print("loading")
        data = json.load(f)
    return data


def generate_meta_train_split(data, ID2CLASS, CLASS2ID):
    JSON_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/"
    image_root = "memcache_manifold://fair_vision_data/tree/coco_train2017"

    data_path = JSON_ANNOTATIONS_DIR
    new_all_cats = list(data["categories"])

    print(f"len of all categories: {len(new_all_cats)}")
    id2img = {i["id"]: i for i in data["images"]}

    anno = defaultdict(list)  # category_id: all annotations
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for c in ID2CLASS.keys():
        img_ids = []
        annotations = []
        for a in anno[c]:
            img_ids.append(a["image_id"])
            annotations.append(a)
        # filter images
        if not visualization:
            img_ids = list(set(img_ids))
        imgs = [id2img[img_id] for img_id in img_ids]
        new_data = {
            "info": data["info"],
            "licenses": data["licenses"],
            "images": imgs,
            "annotations": annotations,
        }
        save_path = get_save_path(data_path, ID2CLASS[c])
        new_data["categories"] = new_all_cats
        with PathManager.open(save_path, "w") as f:
            json.dump(new_data, f)

        # visualize
        if not visualization:
            continue
        dirname = "coco-data-vis" + f"category_{ID2CLASS[c]}"
        os.makedirs(dirname, exist_ok=True)

        for img, anno in zip(imgs, annotations):
            record = {}
            record["file_name"] = os.path.join(image_root, img["file_name"])
            record["height"] = img["height"]
            record["width"] = img["width"]
            image_id = record["image_id"] = img["id"]

            assert (
                anno["image_id"] == image_id
            ), "anno id: {} does not equal to image_id: {}".format(
                anno["image_id"], image_id
            )
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            record["annotations"] = [obj]
            with PathManager.open(record["file_name"], "rb") as f:
                img = np.array(Image.open(f))
            visualizer = Visualizer(img, metadata=None)

            vis = visualizer.draw_dataset_dict(record)
            fpath = os.path.join(dirname, os.path.basename(record["file_name"]))
            vis.save(fpath)


def get_save_path(path, cls):
    prefix = f"category_{cls}"
    save_dir = "coco_meta_learn"
    save_dir = os.path.join(path, save_dir)
    if not PathManager.exists(save_dir):
        PathManager.mkdirs(save_dir)
    save_path = os.path.join(save_dir, prefix + ".json")
    print(f"save_path: {save_path}")
    return save_path


if __name__ == "__main__":
    # check if the category is the same
    ID2CLASS = defaultdict(int)  # id to class
    for items in COCO_CATEGORIES:
        if items["isthing"] == 1:
            ID2CLASS[items["id"]] = items["name"]
    print(len(ID2CLASS), ID2CLASS)
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}
    data = load_train_data()
    generate_meta_train_split(data, ID2CLASS, CLASS2ID)

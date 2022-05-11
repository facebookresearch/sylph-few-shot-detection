#!/usr/bin/env python3
import json
import os
from collections import defaultdict

import detectron2.data.datasets  # noqa # add pre-defined metadata
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO


ann_keys = ["iscrowd", "bbox", "category_id"]
visualization = False

# Novel COCO categories
COCO_NOVEL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
]


def load_train_data():
    annotation_path = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/coco/instances_train2017.json"
    with PathManager.open(annotation_path, "r") as f:
        print("loading")
        data = json.load(f)
    return data


def _read_json_file(json_file):
    json_file = PathManager.get_local_path(json_file, force=True)
    # with contextlib.redirect_stdout(io.StringIO()):
    coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
    return imgs_anns


def generate_meta_train_split(
    data, ID2CLASS, CLASS2ID, base_ids, novel_ids, base_categories, novel_categories
):
    JSON_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/"

    data_path = JSON_ANNOTATIONS_DIR
    new_all_cats = list(data["categories"])
    print(f"len of all categories: {len(new_all_cats)}")
    id2img = {i["id"]: i for i in data["images"]}

    img_id_to_annotations = defaultdict(list)  # image_id: all annotations
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        img_id_to_annotations[a["image_id"]].append(a)

    filtered_image_id_to_annotations_base = []
    filtered_image_id_to_annotations_novel = []
    filtered_image_id_to_annotations_excluded = []
    base_image_ids = set()
    novel_image_ids = set()
    excluded_image_ids = set()
    total_base_image, total_novel_image, excluded_image = 0, 0, 0

    for img_id, anno_list in img_id_to_annotations.items():
        anno_ids = {anno["category_id"] for anno in anno_list}
        if (
            len(anno_ids.intersection(novel_ids)) == 0
        ):  # this image has only annotations from base_ids
            filtered_image_id_to_annotations_base.extend(anno_list)
            total_base_image += 1
            base_image_ids.add(img_id)
        elif (
            len(anno_ids.intersection(base_ids)) == 0
        ):  # this image has only annotations from novel_ids
            filtered_image_id_to_annotations_novel.extend(anno_list)
            total_novel_image += 1
            novel_image_ids.add(img_id)
        else:
            filtered_image_id_to_annotations_excluded.extend(anno_list)
            excluded_image += 1
            excluded_image_ids.add(img_id)
    print(
        f"base images annotations: {len(filtered_image_id_to_annotations_base)}, novel images annotations: {len(filtered_image_id_to_annotations_novel)}, excluded images annotations: {len(filtered_image_id_to_annotations_excluded)}"
    )
    print(
        f"base images: {len(base_image_ids)}, novel images: {len(novel_image_ids)}, excluded images: {len(excluded_image_ids)}"
    )

    # save path:
    save_dir = "coco_meta_learn/coco/"
    save_dir = os.path.join(data_path, save_dir)
    if not PathManager.exists(save_dir):
        PathManager.mkdirs(save_dir)
    # write base
    imgs = [id2img[img_id] for img_id in base_image_ids]
    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": imgs,
        "annotations": filtered_image_id_to_annotations_base,
    }
    new_data["categories"] = [cat for cat in new_all_cats if cat["id"] in base_ids]

    cats = {}
    if "categories" in new_data:
        for cat in new_data["categories"]:
            print(cat)
            cats[cat["id"]] = cat
    print(len(new_data["categories"]))
    print(f"base categories: {new_data['categories']}")
    save_path = os.path.join(save_dir, "base_instances_train2017.json")
    with PathManager.open(save_path, "w") as f:
        json.dump(new_data, f)

    # try to test the json file

    img_annotation = _read_json_file(save_path)
    for img, anno_list in img_annotation:
        print(img, anno_list)
        break
    # write novel
    imgs = [id2img[img_id] for img_id in novel_image_ids]
    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": imgs,
        "annotations": filtered_image_id_to_annotations_novel,
    }
    new_data["categories"] = [cat for cat in new_all_cats if cat["id"] in novel_ids]
    save_path = os.path.join(save_dir, "novel_instances_train2017.json")
    with PathManager.open(save_path, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    # check if the category is the same
    ID2CLASS = defaultdict(int)  # id to class
    for items in COCO_CATEGORIES:
        if items["isthing"] == 1:
            ID2CLASS[items["id"]] = items["name"]
    print(len(ID2CLASS), ID2CLASS)
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}
    data = load_train_data()
    novel_ids = [item["id"] for item in COCO_NOVEL_CATEGORIES]
    novel_categories = [item["name"] for item in COCO_NOVEL_CATEGORIES]
    base_ids = [item["id"] for item in COCO_CATEGORIES if item["id"] not in novel_ids]
    base_categories = [
        item["name"] for item in COCO_CATEGORIES if item["name"] not in novel_categories
    ]
    generate_meta_train_split(
        data, ID2CLASS, CLASS2ID, base_ids, novel_ids, base_categories, novel_categories
    )

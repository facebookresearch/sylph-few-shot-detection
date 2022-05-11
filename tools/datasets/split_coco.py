#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

from detectron2.utils.file_io import PathManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 10], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def load_train_data(args):
    annotation_path = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/coco/instances_train2017.json"
    with PathManager.open(annotation_path, "r") as f:
        print("loading")
        data = json.load(f)
    return data


def generate_seeds(args):
    JSON_ANNOTATIONS_DIR = "manifold://fai4ar/tree/datasets/"

    data_path = JSON_ANNOTATIONS_DIR  # + "coco/instances_train2017.json"
    data = args.data
    new_all_cats = list(data["categories"])

    print(f"new all. cats {new_all_cats}")
    id2img = {i["id"]: i for i in data["images"]}

    anno = defaultdict(list)  # category_id: all images
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in ID2CLASS.keys():
            img_ids = defaultdict(list)  # image id -> all annotations
            for a in anno[c]:
                img_ids[a["image_id"]].append(a)

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10, 30]:
                while True:
                    imgs = random.sample(
                        list(img_ids.keys()), shots
                    )  # sample shot different images for this category
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s["image_id"]:
                                skip = True
                                break
                        if skip:
                            continue
                        if (
                            len(img_ids[img]) + len(sample_shots) > shots
                        ):  # if this image has more annotation needed, skip (not great)
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "images": sample_imgs,  # number of images might be smaller than the number of shots
                    "annotations": sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data["categories"] = new_all_cats
                with PathManager.open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join("cocosplit", "seed" + str(seed))
    save_dir = os.path.join(path, save_dir)
    print(f"save_dir: {save_dir}")
    PathManager.mkdirs(save_dir)
    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
CLASS2ID = {v: k for k, v in ID2CLASS.items()}


class Args:
    seeds = [1, 10]


args = Args()
# args = parse_args()
data = load_train_data(args)
args.data = data
generate_seeds(args)

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

from sylph.utils import create_instances, PathManagerImgLoader, PathManagerImgSave



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="path to the coco_instances_result.json file",  # manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test/20210507200701/eval_pytorch/inference/default/4999/coco_meta_val_novel/
    )
    parser.add_argument(
        "--dataset",
        default="coco_meta_val_novel",
        help="dataset name for dataset dict loading",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="the confidence score for visualization"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    register_all_coco_meta_learn()
    dicts = list(DatasetCatalog.get(args.dataset)[-1])  # get validataion datasets
    print(f"len dicts: {len(dicts)}")
    metadata = MetadataCatalog.get(args.dataset)  # get meta datasets

    # if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

    #     def dataset_id_map(ds_id):
    #         return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    output_dir = args.output_dir
    file = os.path.join(output_dir, "coco_instances_results.json")
    save_path = os.path.join(output_dir, "visualization")

    with PathManager.open(file, "r") as f:
        predictions = json.load(f)
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    index_dict = []
    for ids, dic in enumerate(dicts):
        if ids % 10 == 0:
            print("visualization progress: {}/{}".format(ids, len(dicts)))

        img = PathManagerImgLoader(dic["file_name"])
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.size, metadata, args.conf)
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        output_filename = os.path.join(save_path, basename)
        if not PathManager.exists(save_path):
            PathManager.mkdirs(save_path)
        PathManagerImgSave(Image.fromarray(concat, "RGB"), output_filename)

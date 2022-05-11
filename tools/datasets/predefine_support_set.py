#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy

import detectron2.data.datasets  # noqa # add pre-defined metadata
import numpy as np
from sylph.data.build import get_meta_detection_dataset_dicts
from sylph.data.data_injection.builtin_dataset_few_shot_detection import (
    register_all_coco_meta_learn,
)


def get_support_set(dataset_name, shots=(1, 5, 10), num_seeds=100):
    multi_dataset = get_meta_detection_dataset_dicts([dataset_name], filter_empty=False)

    assert isinstance(
        multi_dataset, dict
    ), f"dataset is not dict, {type(multi_dataset)}, first example, {multi_dataset[0]}"

    metadata = multi_dataset["metadata"]
    classes = deepcopy(metadata["thing_classes"])

    count = 0
    support_set = {}
    for num_shot in shots:
        support_set[f"{num_shot}shot"] = {}
        for seed in range(num_seeds):
            support_set[f"{num_shot}shot"][f"seed{seed}"] = {}
            np.random.seed(seed=seed)
            for class_idx, class_name in enumerate(classes):
                support_set[f"{num_shot}shot"][f"seed{seed}"][
                    class_name
                ] = np.random.choice(
                    multi_dataset[class_idx], num_shot, replace=False
                ).tolist()
                count += 1
                if count % 100 == 0:
                    print(f"progress:{count}/{len(shots)*num_seeds*len(classes)}")

    with open(f"{dataset_name}_support_set.json", "w") as f:
        json.dump(support_set, f)
    print(f"=> saved to {dataset_name}_support_set.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pre-defined support set")
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["coco_meta_val_novel", "coco_meta_val_base"],
    )
    args = parser.parse_args()
    # register
    register_all_coco_meta_learn()
    for dataset_name in args.dataset_names:
        get_support_set(dataset_name)

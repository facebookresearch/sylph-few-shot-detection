#!/usr/bin/env python3
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from sylph.data.data_injection.classes import (
    LVIS_FREQUENT_CATEGORIES,
)
# from sylph.data.data_injection.tfa_builtin_dataset import register_all_tfa_coco_few_shot


def _get_coco_fewshot_instances_meta():
    """
    Default coco meta data
    ret:  from base: {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    Add:
    "novel_dataset_id_to_contiguous_id"
    "novel_classes"
    "base_dataset_id_to_contiguous_id"
    "base_classes"
    """
    meta = {}
    from sylph.data.data_injection.classes import COCO_NOVEL_CLASSES, COCO_BASE_CLASSES

    datasplit_ids = {
        "all": sorted(COCO_NOVEL_CLASSES + COCO_BASE_CLASSES),
        "base": COCO_BASE_CLASSES,
        "novel": COCO_NOVEL_CLASSES,
    }
    for datasplit, ids in datasplit_ids.items():
        datasplit_classes = [k["name"] for k in COCO_CATEGORIES if k["id"] in ids]
        datasplit_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(ids)}
        datasplit_colors = [k["color"] for k in COCO_CATEGORIES if k["id"] in ids]
        meta.update(
            {
                f"{datasplit}_thing_classes": datasplit_classes,
                f"{datasplit}_thing_dataset_id_to_contiguous_id": datasplit_dataset_id_to_contiguous_id,
                f"{datasplit}_thing_colors": datasplit_colors,
            }
        )
    return meta

# def _get_coco_fewshot_instances_meta():
#     ret = _get_coco_instances_meta()
#     novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
#     novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
#     novel_classes = [k["name"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
#     base_categories = [
#         k
#         for k in COCO_CATEGORIES
#         if k["isthing"] == 1 and k["name"] not in novel_classes
#     ]
#     base_ids = [k["id"] for k in base_categories]
#     base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
#     base_classes = [k["name"] for k in base_categories]
#     ret["novel_dataset_id_to_contiguous_id"] = novel_dataset_id_to_contiguous_id
#     ret["novel_classes"] = novel_classes
#     ret["base_dataset_id_to_contiguous_id"] = base_dataset_id_to_contiguous_id
#     ret["base_classes"] = base_classes
#     return ret

def _fewshot_get_builtin_metadata(dataset_name):
    if dataset_name == "coco_meta_learn":
        return _get_coco_fewshot_instances_meta()
    elif dataset_name == "lvis_meta_learn":
        return _get_lvis_fewshot_instances_meta_v1()
    elif dataset_name == "tao_meta_learn":
        return _get_tao_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


def _get_lvis_instances_meta_v1():
    from sylph.data.data_injection.classes import LVIS_V1_CATEGORIES

    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


JSON_ANNOTATIONS_DIR = (
    "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"
)


def _get_lvis_fewshot_instances_meta_v1():
    from sylph.data.data_injection.classes import datasplit_categories

    datasplit_classes = {
        datasplit: [k["name"] for k in categories]
        for datasplit, categories in datasplit_categories.items()
    }
    datasplit_dataset_id_to_contiguous_id = {
        datasplit: {c["id"]: i for i, c in enumerate(categories)}
        for datasplit, categories in datasplit_categories.items()
    }
    # write the meta data
    meta = {}
    for datasplit, classes in datasplit_classes.items():
        meta.update({f"thing_{datasplit}_classes": classes})

    for (
        datasplit,
        dataset_id_to_contiguous_id,
    ) in datasplit_dataset_id_to_contiguous_id.items():
        meta.update(
            {f"{datasplit}_dataset_id_to_contiguous_id": dataset_id_to_contiguous_id}
        )
    return meta


def _get_tao_fewshot_instances_meta():
    """
    similar to _get_lvis_fewshot_instances_meta_v1
    """
    from sylph.data.data_injection.classes import datasplit_categories as lvis_categories

    meta = {}
    datasplit_categories = {
        "base": lvis_categories["basev1"],
        "novel": sorted(LVIS_FREQUENT_CATEGORIES[305:], key=lambda x: x["id"]),  # test
        "novelpartial": sorted(LVIS_FREQUENT_CATEGORIES[305:], key=lambda x: x["id"]),
    }
    datasplit_classes = {
        datasplit: [k["name"] for k in categories]
        for datasplit, categories in datasplit_categories.items()
    }
    datasplit_dataset_id_to_contiguous_id = {
        datasplit: {c["id"]: i for i, c in enumerate(categories)}
        for datasplit, categories in datasplit_categories.items()
    }
    # write the meta data
    meta = {}
    for datasplit, classes in datasplit_classes.items():
        meta.update({f"thing_{datasplit}_classes": classes})
    for (
        datasplit,
        dataset_id_to_contiguous_id,
    ) in datasplit_dataset_id_to_contiguous_id.items():
        meta.update(
            {f"{datasplit}_dataset_id_to_contiguous_id": dataset_id_to_contiguous_id}
        )
    return meta

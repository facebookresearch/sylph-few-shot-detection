import json

from detectron2.utils.file_io import PathManager

COCO_NOVEL_CLASSES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    9,
    16,
    17,
    18,
    19,
    20,
    21,
    44,
    62,
    63,
    64,
    67,
    72,
]
coco_to_synset_path = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/lvis/coco_to_synset.json"


if __name__ == "__main__":
    with PathManager.open(coco_to_synset_path, "r") as f:
        print("loading")
        coco_to_synset = json.load(f)

    LVIS_COCO_NOVEL_OVERLAP_CLSSES = []
    LVIS_COCO_NOVEL_OVERLAP_SYNSET = []

    for _, v in coco_to_synset.items():
        if v["coco_cat_id"] in COCO_NOVEL_CLASSES:
            LVIS_COCO_NOVEL_OVERLAP_CLSSES.append(v["coco_cat_id"])
            LVIS_COCO_NOVEL_OVERLAP_SYNSET.append(v["synset"])

    print(f"LVIS_COCO_NOVEL_OVERLAP_CLSSES: {LVIS_COCO_NOVEL_OVERLAP_CLSSES}")
    print(f"LVIS_COCO_NOVEL_OVERLAP_SYNSET: {LVIS_COCO_NOVEL_OVERLAP_SYNSET}")

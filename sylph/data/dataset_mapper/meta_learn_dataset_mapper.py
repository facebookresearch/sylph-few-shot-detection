"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#!/usr/bin/env python3

import copy
import logging

import numpy as np
import torch
from d2go.data.dataset_mappers import (
    D2GoDatasetMapper,
    D2GO_DATA_MAPPER_REGISTRY,
)
from d2go.data.dataset_mappers.d2go_dataset_mapper import (
    PREFETCHED_SEM_SEG_FILE_NAME,
    read_image_with_prefetch,
)
from detectron2.data import detection_utils as utils, transforms as T

# from d2go.data.fb import detection_utils as d2go_detection_utils

from detectron2.data.transforms.augmentation import (
    AugInput,
    AugmentationList,
)


logger = logging.getLogger(__name__)


@D2GO_DATA_MAPPER_REGISTRY.register()
class MetalearnDatasetMapper(D2GoDatasetMapper):
    """
    Mostly the same with D2GoDatasetMapper, with the change of need_annotation.

    Map a dataset_dict which is a dict from
    "query_set" : 2-d list, c * sq
    "support_set": 2-d list, c * s
    To:
    "query_set" : 1d list, c * sq
    "support_set": 1d list, c * s
    Each item is mapped from annotations to instances, and have the following form:
    {'file_name': 'dataset_path/000000442761.jpg', 'height': 333, 'width': 500, \
        'image_id': 442761, 'image': tensor(), \
        'instances': Instances(num_instances=1, image_height=800, image_width=1201, \
        fields=[gt_boxes: Boxes(tensor([[336.4241, 332.5886, 616.8336, 625.6096]])), gt_classes: tensor([15])])}
    """

    def __init__(
        self, cfg, is_train=True, need_annotation=True, image_loader=None, tfm_gens=None
    ):
        super().__init__(cfg, is_train, image_loader, tfm_gens)
        self.need_annotation = need_annotation

    def _original_call_per_item(self, dataset_dict):
        """
        Modified from d2go's _original_call in D2GoDatasetMapper. The only change is:
        self.need_annotation can happen regarding is_train or not. This is added for
        generating support set while class registration in episodic learning stage
        """
        # new add-on
        # Decouple the two dataset_dicts when in copy-paste mode
        if isinstance(dataset_dict, list):
            assert len(dataset_dict) == 2
            dataset_dict, dataset_dict2 = dataset_dict
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)

        image = self._read_image(dataset_dict, format=self.img_format)

        if not self.backfill_size:
            utils.check_image_size(dataset_dict, image)
        image, dataset_dict = self._custom_transform(image, dataset_dict)

        # Annotation of the dataset dict
        if "annotations" in dataset_dict:
            anno = dataset_dict["annotations"]

        # if self.copy_paste_aug:
        #     # Prepare the second sample for pasting
        # image2 = self._read_image(dataset_dict2, format=self.img_format)
        # if not self.backfill_size:
        #     utils.check_image_size(dataset_dict2, image2)

        # image2, dataset_dict2 = self._custom_transform(
        #     image2, dataset_dict2)

        # # Annotation of the dataset dict
        # anno2 = dataset_dict2["annotations"]

        # # Construct AugDualInput object for CopyPaste augmentation
        # inputs = AugDualInput(
        #     image,
        #     anno=anno,
        #     image_b=image2,
        #     anno_b=anno2,
        # )
        # else:

        inputs = AugInput(image)
        if "annotations" not in dataset_dict:
            transforms = AugmentationList(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens
            )(inputs)
            # # Cache identical transforms in dataset_dict for subclass mappers
            dataset_dict["transforms"] = transforms
            image = inputs.image
        else:
            # pass additional arguments, will only be used when the Augmentation
            #   takes `annotations` as input
            inputs.annotations = dataset_dict["annotations"]
            # Crop around an instance if there are instances in the image.
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                inputs.image = crop_tfm.apply_image(image)

            transforms = AugmentationList(self.tfm_gens)(inputs)
            image = inputs.image
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]
        if image.ndim == 2:
            image = np.expand_dims(image, 2)
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        # Can use uint8 if it turns out to be slow some day
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )
        if not self.need_annotation:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
            # Update the annotations of the background sample (mainly the occlusion info.)
            # Use the d2go version of transform_instances_annotations() to update occlusion
            # label of keypoints and the masks.
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # Transforms the second template sample to paste onto the background
            # if not isinstance(transforms[0], T.NoOpTransform):
            #     # Concatenate the original annotations and the pasted annotations
            #     annos2 = [
            #         utils.transform_instance_annotations(
            #             obj,
            #             transforms[1:],
            #             image_shape,
            #             keypoint_hflip_indices=self.keypoint_hflip_indices,
            #         )
            #         for obj in transforms[0].template_anno
            #         if obj.get("iscrowd", 0) == 0
            #     ]
            #     # Combine the annotations
            #     annos = annos + annos2

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = read_image_with_prefetch(
                dataset_dict.pop("sem_seg_file_name"),
                "L",
                prefetched=dataset_dict.get(
                    PREFETCHED_SEM_SEG_FILE_NAME, None),
            )
            if len(sem_seg_gt.shape) > 2:
                sem_seg_gt = sem_seg_gt.squeeze(2)
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

        # extend standard D2 semantic segmentation to support multiple segmentation
        # files, each file can represent a class
        if "multi_sem_seg_file_names" in dataset_dict:
            raise NotImplementedError()

        if "_post_process_" in dataset_dict:
            proc_func = dataset_dict.pop("_post_process_")
            dataset_dict = proc_func(dataset_dict)
        return dataset_dict

    def _original_call(self, dataset_dict):
        """
        This is added to process an episodic data item where a support set is a list of lists.
        It calles D2GoDatasetMapper._original_call to process an image item,
        and self._original_call_per_item(data) to reserve annotation in test stage for support set
        """
        mapped_dataset_dict = copy.deepcopy(dataset_dict)
        assert isinstance(mapped_dataset_dict, dict)
        if "support_set" not in dataset_dict:
            data = D2GoDatasetMapper._original_call(self, dataset_dict)
            # data = self._original_call_per_item(dataset_dict)
            return data

        if "support_set_target" in dataset_dict:
            mapped_dataset_dict["support_set_target"] = torch.tensor(
                mapped_dataset_dict["support_set_target"]
            )

        # Update  support set and query_set
        # when it is support set, it always needs annotation
        # needs to ensure we always sample enough data even after the jittering.
        if "support_set" in dataset_dict.keys():
            # mapped_dataset_dict["support_set"] = []
            mapped_data_list = []
            numer_of_empty_instances = 0
            for data in dataset_dict["support_set"]:
                mapped_data = self._original_call_per_item(data)
                # print(f"mapped data: {mapped_data}, len instances: {len(mapped_data['instances'])}")
                # non_empty_boxes = mapped_data["instances"].gt_boxes.nonempty(threshold=1e-5)
                if len(mapped_data["instances"]) != 0:
                    mapped_data_list.append(mapped_data)
                else:
                    numer_of_empty_instances += 1
            # print(f"entering support set: {numer_of_empty_instances}")
            if numer_of_empty_instances > 0:
                repeat_data_list = copy.deepcopy(np.random.choice(
                    mapped_data_list, numer_of_empty_instances, replace=True))
                mapped_data_list.extend(repeat_data_list)
            assert len(mapped_data_list) == len(dataset_dict["support_set"])
            mapped_dataset_dict["support_set"] = mapped_data_list
            # print(f"mapped_data_list: {len(mapped_data_list)}")

        if "query_set" in dataset_dict:
            mapped_dataset_dict["query_set"] = []
            for data in dataset_dict["query_set"]:
                mapped_data = D2GoDatasetMapper._original_call(self, data)
                mapped_dataset_dict["query_set"].append(mapped_data)
        # print(f"change with data aug: {mapped_dataset_dict}")
        # map the support_set_target
        return mapped_dataset_dict

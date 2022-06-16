"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import nonzero_tuple
from detectron2.modeling.poolers import (
    ROIPooler,
    convert_boxes_to_pooler_format,
    assign_boxes_to_levels,
)
from detectron2.structures import Instances, Boxes

logger = logging.getLogger(__name__)


def select_a_mask(
    gt_instances: List[Instances], use_all_masks: bool = False
) -> List[Boxes]:

    selected_gt_instances = []
    for per_gt_instances in gt_instances:  # for each image
        boxes_tensor = per_gt_instances.gt_boxes.tensor
        # check empty boxes
        if len(boxes_tensor) == 0:
            logger.info("Run into empty box, use zero masks")
            logger.info(boxes_tensor)
            raise ValueError
        if not use_all_masks:
            # select a single box randomly
            rand_indexes = np.random.choice(range(len(boxes_tensor)), 1)
            box = boxes_tensor[rand_indexes]
            gt_boxes = Boxes(box)
        else:
            gt_boxes = boxes_tensor
        selected_gt_instances.append(gt_boxes)
    return selected_gt_instances


# Modules
class GlobalAdaptiveAvgPool2d(nn.Module):
    """
    Be used to synthesize either kernel(only 1X1) or bias.
    """

    def __init__(self, k_s=1):
        super(GlobalAdaptiveAvgPool2d, self).__init__()
        self.k_s = k_s

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        # H, W = x.data.size(2), x.data.size(3)
        # x = F.avg_pool2d(x, (H, W))
        x = F.adaptive_avg_pool2d(x, (self.k_s, self.k_s))
        x = x.view(N, C, self.k_s, self.k_s)
        return x


class MS_CAM(nn.Module):
    """
    Reimplementation of https://arxiv.org/pdf/2009.14082.pdf
    """

    def __init__(self, channels=64, reduction=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):  # x is pooled feature
        local_context = self.local_att(context)
        global_context = self.global_att(context)
        lg_context = local_context + global_context
        wei = self.sigmoid(lg_context)
        return x * wei


class FeatureFusionModuleV2(nn.Module):
    def __init__(
        self,
        in_channel: int,
        strides: List[int],
        context_attention: bool = False,
        multilevel_feature=False,
    ):
        """
        When context_attention is False, only ROIPooler will be used
        """
        super(FeatureFusionModuleV2, self).__init__()
        self.strides = strides
        self.pooler_output_size = 7
        self.in_channel = in_channel
        self.roi_pooler = MultilevelROIPooler(
            multilevel_feature=multilevel_feature,
            output_size=self.pooler_output_size,
            scales=[1.0 / s for s in self.strides],
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        self.context_attention = context_attention
        if context_attention:
            # reshape features
            self.shape_resize = GlobalAdaptiveAvgPool2d(k_s=self.pooler_output_size)
            # context attention
            self.context_attention_module = MS_CAM(channels=self.in_channel)
            # add additional conv after fusion
            self.conv = nn.Sequential(
                nn.Conv2d(
                    self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1
                ),
                nn.GroupNorm(32, self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features: List[torch.Tensor], bboxes: List[Boxes]):
        """
        Args:
            x: roi pooled feature
        """
        assert features is not None
        device = features[0].device
        # (Bs*num_shots, C, 7, 7)
        pooled_features = self.roi_pooler(features, bboxes)
        # logger.info(f"feature shape: {pooled_features.size()}")
        if self.context_attention:
            # (Bs*num_shots, C, 7, 7)
            pooled_features = self.conv(pooled_features)
            # (Bs*num_shots, C, 7, 7)
            logger.info(f"context shape: {self.shape_resize(features[0]).size()}")
            context = torch.mean(
                torch.stack([self.shape_resize(x).to(device) for x in features]), dim=0
            )
            logger.info(f"context shape: {context.size()}")
            out_features = self.context_attention_module(pooled_features, context)
            return out_features
        else:
            return pooled_features


"""
To export ROIPooler to torchscript, in this file, variables that should be annotated with
`Union[List[Boxes], List[RotatedBoxes]]` are only annotated with `List[Boxes]`.

TODO: Correct these annotations when torchscript support `Union`.
https://github.com/pytorch/pytorch/issues/41412
"""

__all__ = ["MultilevelROIPooler"]


class MultilevelROIPooler(ROIPooler):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        multilevel_feature=False,
        **kwargs,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__(**kwargs)
        self.multilevel_feature = multilevel_feature

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size,
                device=x[0].device,
                dtype=x[0].dtype,
            )
        # (batch_size, 5)
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        if not self.multilevel_feature:
            level_assignments = assign_boxes_to_levels(
                box_lists,
                self.min_level,
                self.max_level,
                self.canonical_box_size,
                self.canonical_level,
            )

            num_boxes = pooler_fmt_boxes.size(0)
            num_channels = x[0].shape[1]
            output_size = self.output_size[0]

            dtype, device = x[0].dtype, x[0].device
            output = torch.zeros(
                (num_boxes, num_channels, output_size, output_size),
                dtype=dtype,
                device=device,
            )

            for level, pooler in enumerate(self.level_poolers):
                inds = nonzero_tuple(level_assignments == level)[0]
                pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
                # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
                output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))
            return output
        else:
            outputs = []
            for level, pooler in enumerate(self.level_poolers):
                output = pooler(x[level], pooler_fmt_boxes)
                outputs.append(output)
            return outputs


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = torch.clamp(
        x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)), min=0.0
    )
    return dist


def _pairwise_squared_euclidean_distance(x):
    """Get the squared euclidean distance between all pairs of vectors in the
    input, as a matrix. The input vectors must be unit length.

    out[i,j] = || x_i - y_j ||_2 ^ 2 in range [0, 4]
    """
    return torch.clamp(2 - 2 * torch.mm(x, x.transpose(0, 1)), min=0.0)


def SoftNearestNeighborLoss(input: torch.Tensor, k: int):
    assert len(input.size()) == 2, f"input has size: {input.size()}"
    # input = standarize(input)
    input = torch.nn.functional.normalize(input)
    dist = _pairwise_squared_euclidean_distance(input)  # 6 \times 6
    # (N, N)
    # dist = pairwise_distances(input) #diagnoal is zero
    dist = torch.exp(-dist)
    intra_class = torch.zeros_like(dist)
    for i in range(dist.size(0)):
        for j in range(dist.size(1)):
            if i // k == j // k and i != j:
                intra_class[i, j] = dist[i, j]

    all_class = torch.zeros_like(dist)
    for i in range(dist.size(0)):
        for j in range(dist.size(1)):
            if i != j:
                all_class[i, j] = dist[i, j]

    per_item = []
    for i in range(dist.size(0)):
        per_item.append(torch.log(intra_class[i].sum() / all_class[i].sum()))

    final = -torch.stack(per_item, dim=0).sum() / dist.size(0)
    return final

from typing import Dict
from collections import defaultdict
from copy import deepcopy
import functools
def convert_list_to_dict(codes: List[Dict]):
    cid_to_class_code_lst = defaultdict(list) # each item is a list of dict
    cid_to_other_field = defaultdict(dict)
    for cls_code in codes:
        cid = cls_code["support_set_target"]
        if torch.is_tensor(cid):
            cid = cid.item()
        assert "class_code" in cls_code
        cid_to_class_code_lst[cid].append(cls_code["class_code"])
        if cid in cid_to_other_field.items():
            # ensure all the field equals
            for key, value in cid_to_other_field[cid]:
                assert cls_code[key] == value, f"key, value pair: {key, value} does not match in {cid_to_other_field}"
        else:
            cid_to_other_field[cid] = deepcopy(cls_code)
            # delete cls code
            del cid_to_other_field[cid]["class_code"]
    return cid_to_class_code_lst, cid_to_other_field

def replace_class_code(support_set_class_code: List[Dict], target_class_codes: List[Dict], device):
    support_set_class_code_cid_dict, support_set_class_code_cid_other_field = convert_list_to_dict(support_set_class_code)
    target_class_codes_cid_dict, target_class_codes_cid_other_field = convert_list_to_dict(target_class_codes)
    overlap_cids = set(support_set_class_code_cid_dict.keys()).intersection(set(target_class_codes_cid_dict.keys()))
    logger.info(f"first set ids: {support_set_class_code_cid_dict.keys()}, second set ids: {target_class_codes_cid_dict.keys()}")
    logger.info(f"overlap_cids: {overlap_cids}, len: {len(overlap_cids)}")
    results = deepcopy(support_set_class_code)
    for result in results:
        cid = result["support_set_target"]
        if torch.is_tensor(cid):
            cid = cid.item()
        # result["class_code"]=result["class_code"][0]
        if cid in overlap_cids:
            logger.info(f"replace cid: {cid}")
            result["class_code"] = target_class_codes_cid_dict[cid][0]
        # put target to device
        for k, v in result["class_code"].items():
            result["class_code"][k] = torch.tensor(v).to(device)
    return results


def reduce_class_code(out_codes: List[Dict]):
    # reduce
    if len(out_codes) == 0:
        return out_codes

    assert "class_code" in out_codes[0]
    all_keys = out_codes[0]["class_code"].keys()
    cid_to_class_code_lst, cid_to_other_field = convert_list_to_dict(out_codes)
    logger.info(f"before reduce class code: {out_codes[0]['class_code']['cls_conv'].size()}, {out_codes[0]['class_code']['cls_bias'].size()}")

    # reduce a list to a single sum
    results = []
    for cid, code_lst in cid_to_class_code_lst.items():
        result = cid_to_other_field[cid]
        result["class_code"] = {key: functools.reduce(lambda x, y: x + y[key], code_lst, 0) for key in all_keys}
        acc_weight = result["class_code"]["acc_weight"]
        if abs(1.0-acc_weight) > 1e-6:
            acc_weight = result["class_code"]["acc_weight"]
            logger.info(f"category id: {cid}, is using only {acc_weight}, rebalance it")
            result["class_code"]["cls_conv"] = result["class_code"]["cls_conv"]/acc_weight
            result["class_code"]["cls_bias"] = result["class_code"]["cls_bias"]/acc_weight
            if "cls_weight_norm" in result["class_code"]:
                result["class_code"]["cls_weight_norm"] = result["class_code"]["cls_weight_norm"]/acc_weight
            result["class_code"]["acc_weight"] = 1.0
        # delete acc weight to be consistent
        del result["class_code"]["acc_weight"]
        # l2 normalize the code
        # result["class_code"]["cls_conv"] = F.normalize(result["class_code"]["cls_conv"], p=2, dim=1)
        results.append(result)
    logger.info(f"after reduce class code: {results[0]['class_code']['cls_conv'].size()}, {results[0]['class_code']['cls_bias'].size()}, {results[0]['class_code']['cls_weight_norm'].size()}")
    return results

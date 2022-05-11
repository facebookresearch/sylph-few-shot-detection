#!/usr/bin/env python3
import logging
from typing import List, Tuple, Dict, Optional  # noqa

import torch
import torch.nn.functional as F
from adet.layers import ml_nms
from sylph.modeling.meta_fcos.iou_loss import IOULoss
from adet.utils.comm import reduce_sum
from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
    )
    return torch.sqrt(ctrness)


class FCOSOutputs(nn.Module):
    def __init__(self, cfg):
        super(FCOSOutputs, self).__init__()
        self.cfg = cfg
        self._init_fcos()
        self._init_code_generator()

    def _init_fcos(self):
        # Get fcos configs
        self.focal_loss_alpha = self.cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = self.cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample = self.cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = self.cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = self.cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = self.cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = self.cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(self.cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        self.pre_nms_thresh_test = self.cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_test = self.cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = self.cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.nms_thresh = self.cfg.MODEL.FCOS.NMS_TH
        self.thresh_with_ctr = self.cfg.MODEL.FCOS.THRESH_WITH_CTR
        self.strides = self.cfg.MODEL.FCOS.FPN_STRIDES
        self.box_branch_loss_on = (
            False
            if self.cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_BBOX_BRANCH
            or self.cfg.MODEL.PROPOSAL_GENERATOR.FREEZE
            else True
        )

        # Generate sizes of interest
        SIZES_OF_INTEREST = self.cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.sizes_of_interest = [
            [x0, x1]
            for x0, x1 in zip([-1] + SIZES_OF_INTEREST, SIZES_OF_INTEREST + [INF])
        ]
        self.num_classes = (
            self.cfg.MODEL.FCOS.NUM_CLASSES
        )  # non episodic learning, label range from 0, num_classes
        self.back_ground_id = 100000  # a number >> num_classes, so that we support a dynamic number of classes
        self.box_quality = sorted(self.cfg.MODEL.FCOS.BOX_QUALITY) #'iou', 'ctrness'

    def _init_code_generator(self):
        # Episodic learning
        self.episodic_learning = self.cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        if not self.episodic_learning:
            return
        self.code_generator_name = self.cfg.MODEL.META_LEARN.CODE_GENERATOR.NAME
        # all shared code_generator configs
        self.distillation_loss_weight = (
            self.cfg.MODEL.META_LEARN.CODE_GENERATOR.DISTILLATION_LOSS_WEIGHT
        )
        self.distill_loss_type = nn.L1Loss(reduction="mean")
        self.query_shot = self.cfg.MODEL.META_LEARN.QUERY_SHOT
        if self.code_generator_name == "CodeGenerator":
            # get code generator config if it is meta-training stage
            self.box_on = self.cfg.MODEL.META_LEARN.CODE_GENERATOR.BOX_ON
        elif self.code_generator_name == "ROIEncoder":
            self.box_on = False
        else:
            raise ValueError(f"{self.code_generator_name} is not supported")

    def _transpose(self, training_targets, num_loc_list):
        """
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _get_ground_truth(
        self,
        locations: List[torch.Tensor],
        gt_instances: List[Instances],
    ):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for level, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(
                self.sizes_of_interest[level]
            )
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[level], -1)
            )
        # shape (K, 2)
        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        # shape (K, 2) where K = \sum_i Ri = \sum_i Hi * Wi
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )
        # Create a list of locations and image index for each image in the batch.
        # Each list item has shape (K, 2)
        training_targets["locations"] = [
            locations.clone() for _ in range(len(gt_instances))
        ]
        # Each list item has shape (K,)
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i
            for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        # shape (B, R_i, 4)
        reg_targets = training_targets["reg_targets"]
        for idx in range(len(reg_targets)):
            reg_targets[idx] = reg_targets[idx] / float(self.strides[idx])

        return training_targets

    def get_sample_region(
        self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1
    ):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax
            )
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(
        self,
        locations: torch.Tensor,
        targets: List[Instances],
        size_ranges: torch.Tensor,
        num_loc_list: List[int],
    ):
        """
        Computes the following results as a list, with length/; K = \sum_i Hi * Wi, each item represents a location :
        1. "reg_targets" regression targets: includes l, t, r, b
        2. "labels", each bounding box has a matching label (continuguois id, else background id)
        3. "target_inds": which target in the box gt of image i
        """
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(
                    labels_per_im.new_zeros(locations.size(0)) + self.back_ground_id
                )
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue
            #what is the area here?
            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            # shape (K, N_i, 4) where K = \sum_i Hi * Wi and N_i is no. of gt bbox in image i
            # one pixel at k, will be matched to maximum N_i targets
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:  # use logics here
                    bitmasks = None
                # shape (K, N_i)
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    num_loc_list,
                    xs,
                    ys,
                    bitmasks=bitmasks,
                    radius=self.radius,
                )
            else:
                # only the minimum of l,t,r,b is larger than 0, then it is in the box
                # (K, N_i)
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # shape (K, N_i)
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            # shape (K, N_i)
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[:, [0]]) & (
                max_reg_targets_per_im <= size_ranges[:, [1]]
            )
            # shape (K, N_i)
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            # shape (K,), shape (K,)
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(
                dim=1
            )
            # shape (K, 4), index in N_i
            reg_targets_per_im = reg_targets_per_im[
                range(len(locations)), locations_to_gt_inds
            ]
            # shape (K, ), points to which gt box in image i
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)  # total gt instances
            # shape (K, )
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.back_ground_id

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds,
        }

    def losses(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        iou_pred,
        locations,
        gt_instances,
        top_feats=None,
        support_set_targets: List[torch.Tensor] = None,
        support_set_per_class_code: Dict = None,
        cls_logits_kernel: Tuple = None,
    ):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        # shape (B*K,)
        instances.labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["labels"]
            ],
            dim=0,
        )
        # shape (B*K,)
        instances.gt_inds = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["target_inds"]
            ],
            dim=0,
        )
        # shape (B*K,)
        instances.im_inds = cat(
            [x.reshape(-1) for x in training_targets["im_inds"]], dim=0
        )
        # shape (B*K, 4)
        instances.reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4)
                for x in training_targets["reg_targets"]
            ],
            dim=0,
        )
        # shape (B*K, 2)
        instances.locations = cat(
            [x.reshape(-1, 2) for x in training_targets["locations"]], dim=0
        )
        # shape (B*L,) where L is total FPN levels
        instances.fpn_levels = cat(
            [x.reshape(-1) for x in training_targets["fpn_levels"]], dim=0
        )
        # shape (B*K, C)
        instances.logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(
                    -1, x.size(1)
                )  # output using the dimension of C
                for x in logits_pred
            ],
            dim=0,
        )
        B = reg_pred[0].size(1)
        # shape (B*K, 4)
        instances.reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B) # B cant be class * 4
                x.permute(0, 2, 3, 1).reshape(-1, B)
                for x in reg_pred
            ],
            dim=0,
        )
        # shape (B*K, 1)
        instances.ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.permute(0, 2, 3, 1).reshape(-1)
                for x in ctrness_pred
            ],
            dim=0,
        )
        # shape (B*K, 1)
        instances.iou_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.permute(0, 2, 3, 1).reshape(-1)
                for x in iou_pred
            ],
            dim=0,
        )

        if len(top_feats) > 0:
            instances.top_feats = cat(
                [
                    # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                    x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                    for x in top_feats
                ],
                dim=0,
            )

        if self.episodic_learning:
            assert support_set_targets is not None
            logger.debug(f"support_set_targets: {support_set_targets}")
            # cat support_set
            support_set_target_to_index = {
                tid.item(): cid for cid, tid in enumerate(support_set_targets)
            }
            support_set_target_class_indices = [
                tid.item() for tid in support_set_targets
            ]
            support_set_target_to_index.update(
                {self.num_classes: len(support_set_targets)}
            )
            # shape (1, C) where C is no. of object classes
            support_set_targets = torch.stack(support_set_targets, dim=0).view(1, -1)
            # shape (Bs*K, C) eg. (34, 56, 89) # continuous id already
            support_set_targets = support_set_targets.repeat(
                instances.labels.size(0), 1
            )
            return self.fcos_losses_episodic_learning(
                instances,
                support_set_targets,
                support_set_target_to_index,
                support_set_target_class_indices,
                support_set_per_class_code,
                cls_logits_kernel,
            )
        return self.fcos_losses(instances)

    def fcos_losses_episodic_learning(
        self,
        instances,
        support_set_targets,
        support_set_target_to_index,
        support_set_target_class_indices,
        support_set_per_class_code=None,
        cls_logits_kernel=None,
    ):
        # 3's gt_loc is 0000
        # n * 1 1*5
        labels = instances.labels.unsqueeze(dim=-1).repeat(
            1, support_set_targets.size(1)
        )
        assert (
            labels.size() == support_set_targets.size()
        ), f"{labels.size()} != {support_set_targets.size()}"
        # (N*C, 1)- > (N*C, 1)

        # returns index of the nonzero value, non background
        pos_inds = torch.nonzero(
            instances.labels.flatten() != self.back_ground_id
        ).squeeze(1)

        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        # [num_locations, c]
        class_target = (support_set_targets == labels).to(torch.float)

        class_loss = (
            sigmoid_focal_loss_jit(
                instances.logits_pred,
                class_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / num_pos_avg
        )

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds
        if self.box_on:  # adapt the box
            continous_labels = torch.tensor(
                [support_set_target_to_index[l.item()] for l in instances.labels]
            ).to(
                support_set_targets.device
            )  # 0, 1,2, 3
            N, C = instances.reg_pred.size()
            pred = instances.reg_pred.view(N, int(C // 4), 4)  # N, C, 4
            target_cls = continous_labels.view(-1, 1, 1).expand(-1, -1, 4)
            pred = torch.gather(pred, 1, target_cls).squeeze(1)
            instances.reg_pred = pred

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)

        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        ious, gious = self.loc_loss_func.compute_ious(instances.reg_pred, instances.reg_targets)

        if pos_inds.numel() > 0:
            if not self.box_on:
                reg_loss = (
                    self.loc_loss_func(
                        ious, gious, ctrness_targets
                    )
                    / loss_denorm
                )
            else:
                reg_loss = (
                    self.loc_loss_func(
                        ious,
                        gious,
                        ctrness_targets,
                    )
                    / loss_denorm
                )

            ctrness_loss = (
                F.binary_cross_entropy_with_logits(
                    instances.ctrness_pred, ctrness_targets, reduction="sum"
                )
                / num_pos_avg
            )
        else:
            reg_loss = instances.reg_pred.sum() * 0
            ctrness_loss = instances.ctrness_pred.sum() * 0

        losses = {
            "loss_fcos_cls": class_loss,
        }

        # distilattion loss
        if cls_logits_kernel is not None and self.distillation_loss_weight > 0:
            cls_logits_weight, cls_logits_bias = cls_logits_kernel
            support_set_per_class_conv = support_set_per_class_code["cls_conv"]
            bias = support_set_per_class_code["cls_bias"]

            # get the target class code from pretrained conv kernel
            support_set_target_class_indices = torch.tensor(
                support_set_target_class_indices
            ).to(cls_logits_weight.device)
            support_set_per_class_code_target = torch.index_select(
                cls_logits_weight, 0, support_set_target_class_indices
            )
            bias_target = torch.index_select(
                cls_logits_bias, 0, support_set_target_class_indices
            )
            # ensure the generated class code shall have the same shape with the original (pretrained) one
            assert (
                support_set_per_class_conv.shape
                == support_set_per_class_code_target.shape
            ), f"Got {support_set_per_class_conv.shape} x {support_set_per_class_code_target.shape}"

            assert (
                bias.shape == bias_target.shape
            ), f"Got {bias.shape} x {bias_target.shape}"
            distillation_loss = (
                self.distill_loss_type(
                    support_set_per_class_conv,
                    support_set_per_class_code_target,
                )
                + self.distill_loss_type(bias, bias_target)
            ) * self.distillation_loss_weight
            losses.update({"loss_gen_distill": distillation_loss})

        if self.box_branch_loss_on:
            losses.update(
                {
                    "loss_fcos_loc": reg_loss,
                    "loss_fcos_ctr": ctrness_loss,
                }
            )

        extras = {"instances": instances, "loss_denorm": loss_denorm}
        return extras, losses

    def fcos_losses(self, instances):
        losses, extras = {}, {}

        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()  # (N, 1, Hi, Wi) -> (N*Hi*Wi,)

        pos_inds = torch.nonzero(labels != self.back_ground_id).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = (
            sigmoid_focal_loss_jit(
                instances.logits_pred,
                class_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / num_pos_avg
        )

        if self.cfg.MODEL.PROPOSAL_GENERATOR.OWD:
            class_loss.detach()
        elif self.cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_LOGITS and self.cfg.MODEL.PROPOSAL_GENERATOR.FREEZE_CLS_LOGITS:
            class_loss.detach()
        else:
            losses["loss_fcos_cls"] = class_loss

        # 2. compute the box regression and quality loss with FOREGROUND SAMPLES
        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        iou_foreground_targets, giou_foreground_targets = self.loc_loss_func.compute_ious(instances.reg_pred, instances.reg_targets)
        if self.cfg.MODEL.FCOS.IOU_MASK:
            iou_foreground_targets[iou_foreground_targets < 0.3] = 0

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        ctrness_loss = F.binary_cross_entropy_with_logits(
            instances.ctrness_pred, ctrness_targets.clone().detach(),
            reduction="sum"
        ) / num_pos_avg

        iou_quality_loss = F.binary_cross_entropy_with_logits(
            instances.iou_pred, iou_foreground_targets.clone().detach(),
            reduction="sum"
        ) / num_pos_avg

        if self.box_quality == ['ctrness', 'iou']:
            reg_loss = self.loc_loss_func(
                iou_foreground_targets,
                giou_foreground_targets,
                weight=ctrness_targets
            ) / loss_denorm
            if self.box_branch_loss_on:
                losses['loss_fcos_iou'] = iou_quality_loss
                losses['loss_fcos_ctr'] = ctrness_loss
                losses['loss_fcos_loc'] = reg_loss
            else:
                iou_quality_loss.detach()
                ctrness_loss.detach()
                reg_loss.detach()
        elif self.box_quality == ['ctrness']:
            reg_loss = self.loc_loss_func(
                iou_foreground_targets,
                giou_foreground_targets,
                weight=ctrness_targets
            ) / loss_denorm
            iou_quality_loss.detach()
            if self.box_branch_loss_on:
                losses['loss_fcos_ctr'] = ctrness_loss
                losses['loss_fcos_loc'] = reg_loss
            else:
                ctrness_loss.detach()
                reg_loss.detach()
        elif self.box_quality == ['iou']:
            reg_loss = self.loc_loss_func(
                iou_foreground_targets,
                giou_foreground_targets,
            ) / num_pos_avg
            ctrness_loss.detach()
            if self.box_branch_loss_on:
                losses['loss_fcos_iou'] = iou_quality_loss
                losses['loss_fcos_loc'] = reg_loss
            else:
                iou_quality_loss.detach()
                reg_loss.detach()
        else:
            raise NotImplementedError

        extras = {"instances": instances, "loss_denorm": loss_denorm}
        return extras, losses

    def predict_proposals(
        self,
        logits_pred: List[torch.Tensor],
        reg_pred: List[torch.Tensor],
        ctrness_pred: List[torch.Tensor],
        iou_pred: List[torch.Tensor],
        locations: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        top_feats: List[torch.Tensor],
    ):
        """
        Used mostly in testing stage to make proposals
        """
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations,
            "o": logits_pred,
            "r": reg_pred,
            "c": ctrness_pred,
            "i": iou_pred,
            "s": self.strides,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for level, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            i = per_bundle["i"]
            t = per_bundle["t"] if "t" in bundle else None

            # if self.box_on:
            #     sampled_boxes.append(
            #         self.forward_for_single_feature_map_box_per_class(
            #             l, o, r, c, image_sizes, t
            #         )
            #     )

            # else:
            sampled_boxes.append(
                self.forward_for_single_feature_map(l, o, r, c, i, image_sizes, t)
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = (
                    l.new_ones(len(per_im_sampled_boxes), dtype=torch.long) * level
                )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map_box_per_class(
        self, locations, logits_pred, reg_pred, ctrness_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()

        # Single-label prediction: for each pixel, only choose the category that has the largest prediction
        value, index = logits_pred.max(dim=-1, keepdim=True)
        logits_pred = F.one_hot(index.squeeze(dim=-1), num_classes=C) * value
        num_class = int(reg_pred.size(1) // 4)
        # b, c X4, h, w
        box_regression = reg_pred.view(N, 4, num_class, H, W).permute(0, 3, 4, 2, 1)
        box_regression = box_regression.reshape(N, -1, num_class, 4)  # N, H*W, C, 4
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        # for each image
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            # indexes
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]  # L
            per_class = per_candidate_nonzeros[:, 1]  # L

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]  # L, C, 4
            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            # select by class label
            target_cls = torch.tensor(per_class).view(-1, 1, 1).expand(-1, -1, 4)
            per_box_regression = torch.gather(
                per_box_regression, 1, target_cls
            ).squeeze(1)
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(
                    per_pre_nms_top_n, sorted=False
                )
                per_class = per_class[top_k_indices]  # 100
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack(
                [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ],
                dim=1,
            )

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def forward_for_single_feature_map(
        self, locations, logits_pred, reg_pred, ctrness_pred, iou_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C)

        if self.cfg.MODEL.PROPOSAL_GENERATOR.OWD:
            logits_pred = torch.ones_like(logits_pred)
            logits_pred = logits_pred[:, :, [0]] # only one class in OWD
        else:
            #logits_pred = torch.ones_like(logits_pred)
            logits_pred = logits_pred.sigmoid()
        # print(f"logits_pred: {logits_pred}")

        # for each pixel, only choose the category that has the largest prediction
        # value, index = logits_pred.max(dim=-1, keepdim=True)
        # logits_pred = F.one_hot(index.squeeze(dim=-1), num_classes=C) * value
        # b, c X4, h, w
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        iou_pred = iou_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        iou_pred = iou_pred.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness/iou scores before applying the threshold.
        if self.thresh_with_ctr or self.cfg.MODEL.PROPOSAL_GENERATOR.OWD:
            if self.box_quality == ['ctrness']:
                logits_pred = logits_pred * ctrness_pred[:, :, None]
            elif self.box_quality == ['iou']:
                logits_pred = logits_pred * iou_pred[:, :, None]
            elif self.box_quality == ['ctrness', 'iou']:
                logits_pred = logits_pred * torch.sqrt(iou_pred[:, :, None]*ctrness_pred[:, :, None])
            else:
                raise NotImplementedError()

        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr and not self.cfg.MODEL.PROPOSAL_GENERATOR.OWD:
            if self.box_quality == ['ctrness']:
                logits_pred = logits_pred * ctrness_pred[:, :, None]
            elif self.box_quality == ['iou']:
                logits_pred = logits_pred * iou_pred[:, :, None]
            elif self.box_quality == ['ctrness', 'iou']:
                logits_pred = logits_pred * torch.sqrt(iou_pred[:, :, None]*ctrness_pred[:, :, None])
            else:
                raise NotImplementedError()

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(
                    per_pre_nms_top_n, sorted=False
                )
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack(
                [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ],
                dim=1,
            )

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(), number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

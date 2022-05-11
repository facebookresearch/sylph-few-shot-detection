# Copyright (c) Facebook, Inc. and its affiliates.
#!/usr/bin/env python3
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.fb.legacy import FastRCNNOutputs
from detectron2.layers import nonzero_tuple, Linear
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F

from .head_utils import (
    LinearModule,
    sigmoid_focal_loss_with_mask,
    smooth_l1_loss_with_weight,
    extract_mask,
)

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


"""
FocalLossOutputs and FocalLossOutputsLayers
taken from ~/fbsource/fbcode/fblearner/flow/projects/vision/xray_detection_beta/modeling/roi_heads/xray_box_predictor.py
"""


class FocalLossOutputs(FastRCNNOutputs):
    """
    It inherits FastRCNNOutputs, a class providing methods used to decode the
    outputs of a Fast R-CNN head, used in `FastRCNNOutputLayers`, the `box_predictor`
    of `detectron2.modeling.roi_heads.roi_heads.StandardROIHeads`.match_pairs.device
    Both return a dictionary in function `losses()`. The difference is that  "loss_cls"
    is calculated by `sigmoid_focal_loss` instead of `softmax_cross_entropy_loss`.
    """

    def __init__(
        self, focal_loss_alpha=0.5, focal_loss_gamma=0.0, score_threshold=0.05, **kwargs
    ):
        """
        Args:
            One change in **kwargs:
            pred_class_logits (Tensor): A tensor of shape (R, K) instead of (R, K + 1)
                storing the predicted class logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
                (Reason for shape change: Logits for background is no longer predicted
                since sigmoid is used instead of softmax and so "background" can be
                reflected by low logits for all foregrounds. )
        """
        super().__init__(**kwargs)
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.score_threshold = score_threshold
        self.num_classes = self.pred_class_logits.size(1)
        # Pre-calculated the below to make the overwriting of multi-label cleaner

        self.mask_info_carrier = kwargs["proposals"]

        if hasattr(self, "gt_boxes"):
            self.box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
            self.cls_agnostic_bbox_reg = (
                self.pred_proposal_deltas.size(1) == self.box_dim
            )

        # `gt_classes` is zero-indexed; [num_classes] means a proposal is background
        if hasattr(self, "gt_classes"):
            self.gt_labels_target = F.one_hot(
                self.gt_classes, num_classes=self.num_classes + 1
            )[:, :-1].to(self.pred_class_logits.dtype)
        # no loss for the last (background) class
        # If a proposal is background, targets will be all zero at the K indices.

        self.reg_loss_weight = None

    def sigmoid_focal_loss(self):
        """
        Compute the (sigmoid) focal loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()

            focal_loss = sigmoid_focal_loss_with_mask(
                self.pred_class_logits,
                self.gt_labels_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
                mask=extract_mask(self.mask_info_carrier, self.num_classes).to(
                    self.pred_class_logits.device
                ),
            )
            return focal_loss / self.gt_labels_target.size(0)

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression. Similar to
        `FastRCNNOutputs.box_reg_loss()`, except the way to get `bg_class_ind`.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        device = self.pred_proposal_deltas.device

        bg_class_ind = self.num_classes

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on
        # predictions for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        )[0]
        if self.cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(self.box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns
            # [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = self.box_dim * fg_gt_classes[:, None] + torch.arange(
                self.box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss_with_weight(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
                weight=self.reg_loss_weight,
            )
        # exclude "giou"
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / (
            self.gt_classes.numel()
            if self.reg_loss_weight is None
            else self.reg_loss_weight.sum()
        )
        return loss_box_reg

    def losses(self):
        return {
            "loss_cls": self.sigmoid_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def _log_accuracy(self):
        """
        Log the accuracy metrics in training to EventStorage. Similar to
        `FastRCNNOutputs._log_accuracy`, except
          (1) how to get `bg_class_ind`;
          (2) when to predict background without the corresponding logit
        """
        num_instances = self.gt_classes.numel()
        pred_scores, pred_classes = self.pred_class_logits.max(dim=1)
        bg_class_ind = self.num_classes
        # Without the logits specific to background, we predict a proposal to be
        # background if all of its foreground logits is under the threshold.
        pred_bg_mask = pred_scores.sigmoid() < self.score_threshold
        pred_classes[pred_bg_mask] = bg_class_ind

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar(
                    "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
                )
                storage.put_scalar(
                    "fast_rcnn/false_negative", num_false_negative / num_fg
                )


class FocalLossOutputLayers(FastRCNNOutputLayers):
    """
    It inherits FastRCNNOutputLayers, the `box_predictor` of
    `detectron2.modeling.roi_heads.roi_heads.StandardROIHeads`
    They both include two linear layers:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    The difference is that
        - in `losses()`, it uses `FocalLossOutputs()` instead of
          `detectron2.modeling.roi_heads.roi_heads.fast_rcnn.FastRCNNOutputs()`.
          Then, classification scores here are changed from cross entropy to focal loss.
        - in `inference()`, we use sigmoid instead of softmax to predict probability.
    """

    @configurable
    def __init__(
        self,
        *,
        focal_loss_alpha: float = 0.5,
        focal_loss_gamma: float = 0.0,
        train_score_thresh: float = 0.05,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            focal_loss_alpha (float): between 0 and 1, or something negative
            focal_loss_gamma (float):
        """
        super().__init__(**kwargs)

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # It was defined in FastRCNNOutputLayers:
        #         self.cls_score = Linear(input_size, num_classes + 1)
        # Logits for background is no longer predicted since sigmoid is used instead of
        # softmax and "background" can be reflected by low logits for all foregrounds.
        self.cls_score = Linear(self.cls_score.in_features, kwargs["num_classes"])
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        # `self.cls_score` will be used in `forward()` by: `scores = self.cls_score(x)`
        # The output of `forward()` is a tuple `(scores, _)`, which will be passed to
        # `losses()` and `inference()` as one of the arguments `predictions[tuple]`.

        self.train_score_thresh = train_score_thresh

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["focal_loss_alpha"] = cfg.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_ALPHA
        ret["focal_loss_gamma"] = cfg.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_GAMMA
        ret["loss_weight"]["loss_cls"] = cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT
        ret["train_score_thresh"] = cfg.MODEL.ROI_BOX_HEAD.SCORE_THRESH_TRAIN
        return ret

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.

        Returns:
            dict[str->float]: with keys "loss_cls", "loss_box_reg"
        """
        scores, proposal_deltas = predictions
        losses = FocalLossOutputs(
            focal_loss_alpha=self.focal_loss_alpha,
            focal_loss_gamma=self.focal_loss_gamma,
            score_threshold=self.train_score_thresh,
            box2box_transform=self.box2box_transform,
            pred_class_logits=scores,
            pred_proposal_deltas=proposal_deltas,
            proposals=proposals,
            smooth_l1_beta=self.smooth_l1_beta,
            box_reg_loss_type=self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def predict_probs(self, predictions, proposals):
        """
        This function will be called in `FastRCNNOutputLayers.inference()`.
        The main difference from `FastRCNNOutputLayers.predict_probs()`:
          - The shape of `scores` in `predictions` is changed: no background logits.
          - The probability is calculated with sigmoid instead of softmax.
        The Returns will be passed into `fast_rcnn_inference` defined in
        `detectron2/modeling/roi_heads/fast_rcnn.py`. So, the last index for background
        will be added to expand `probs` from K to K+1 in dim 2. That last index will
        actually be simply ignored in the following inference functions.

        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]: Tensor List of predicted class probabilities for each image.
                Element i has shape (Ri, K+1). Ri is the proposal number for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        scores = scores.sigmoid_()
        # The following inference function is implemented to ignore the placeholder.
        background_placeholder = torch.zeros(scores.shape[0], 1, device=scores.device)
        probs = torch.cat((scores, background_placeholder), 1)
        return probs.split(num_inst_per_image, dim=0)


"""
Our Output layers inheriting FocalLossOutputLayers and FocalLossOutput
"""


class BiFocalLossOutputLayers(FocalLossOutputLayers):
    """
    BiFocalLossOutputLayers support both few shots and many shots loss and outputs,
    it  calls FocalLossOutputs for many shots
    and FewShotFocalLossOutputs for few shots
    Change loss
    """

    @configurable
    def __init__(
        self,
        *,
        cfg,
        episodic_learning: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            focal_loss_alpha (float): between 0 and 1, or something negative
            focal_loss_gamma (float):
        """
        super().__init__(**kwargs)
        self.episodic_learning = episodic_learning
        self.cfg = cfg

        if self.episodic_learning:
            # add cond clas_layer here
            self.cls_score = LinearModule()
        else:
            self.cls_score = Linear(self.cls_score.in_features, kwargs["num_classes"])
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["cfg"] = cfg
        ret["episodic_learning"] = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        return ret

    def forward(
        self, x: torch.Tensor, class_codes: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
            class_codes: in training will be C

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if not self.episodic_learning:
            return super().forward(x)
        # x: (N, 1024), N number of proposals
        assert "cls_conv" in class_codes
        assert "cls_bias" in class_codes
        assert len(class_codes["cls_conv"].size()) == 4
        assert (
            class_codes["cls_conv"].size(2) == 1
            and class_codes["cls_conv"].size(3) == 1
        )

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if self.episodic_learning:
            assert class_codes is not None
            scores = self.cls_score(
                x,
                class_codes["cls_conv"].view(-1, class_codes["cls_conv"].size(1)),
                class_codes["cls_bias"],
            )
        else:
            scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(
        self,
        predictions,
        proposals: List[Instances],
        support_set_target: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.

        Returns:
            dict[str->float]: with keys "loss_cls", "loss_box_reg"
        """
        # pretraining use FocalLossOutputs
        if not self.episodic_learning:
            return super().losses(predictions, proposals)
        # meta-learning loss
        scores, proposal_deltas = predictions

        losses = FewShotFocalLossOutputs(
            support_set_target=support_set_target,
            total_num_classes=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            focal_loss_alpha=self.focal_loss_alpha,
            focal_loss_gamma=self.focal_loss_gamma,
            score_threshold=self.train_score_thresh,
            box2box_transform=self.box2box_transform,
            pred_class_logits=scores,
            pred_proposal_deltas=proposal_deltas,
            proposals=proposals,
            smooth_l1_beta=self.smooth_l1_beta,
            box_reg_loss_type=self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


class FewShotFocalLossOutputs(FocalLossOutputs):
    """
    Adapted from FocalLossOutputs.
    """

    def __init__(
        self,
        support_set_target: List[torch.Tensor],
        total_num_classes: int,
        focal_loss_alpha: float = 0.5,
        focal_loss_gamma: float = 0.0,
        score_threshold: float = 0.05,
        **kwargs,
    ):
        """
        Adapted from FocalLossOutputs.__init__, adding two args:
            support_set_target: provides the real class_id
            total_num_classes: provides the total num classes in the training dataset
        """
        FastRCNNOutputs.__init__(self, **kwargs)
        # FocalLossOutputs's init
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.score_threshold = score_threshold

        # Pre-calculated the below to make the overwriting of multi-label cleaner

        self.mask_info_carrier = kwargs["proposals"]

        if hasattr(self, "gt_boxes"):
            self.box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
            self.cls_agnostic_bbox_reg = (
                self.pred_proposal_deltas.size(1) == self.box_dim
            )
        self.reg_loss_weight = None
        # CHANGE START
        # Dynamic setup on num_classes
        self.num_classes = self.pred_class_logits.size(1)
        # print(f"self.num_classes: {self.num_classes}, support_set_target: {support_set_target}")
        assert self.num_classes == len(support_set_target)
        self.gt_to_cid = {
            id.item(): int(cid) for cid, id in enumerate(support_set_target)
        }
        self.gt_to_cid[total_num_classes] = len(
            support_set_target
        )  # replace the background id

        # `gt_classes` is zero-indexed; [num_classes] means a proposal is background
        if hasattr(self, "gt_classes"):
            # print(f" have gt_class: {self.gt_classes}") # num_classes from config is the background (depends on the training dataset)
            # convert gt_class's to continuous id in a small task
            self.gt_classes = torch.tensor(
                [self.gt_to_cid[id.item()] for id in self.gt_classes]
            ).to(self.pred_class_logits.device)
            # print(f" have gt_class: {self.gt_classes}")
            self.gt_labels_target = F.one_hot(
                self.gt_classes, num_classes=self.num_classes + 1
            )[:, :-1].to(self.pred_class_logits.dtype)
            # print(f" have gt_targers: {self.gt_labels_target}")


class MappedInstances(Instances):
    """
    Wrapper around `detectron2.structures.instances.Instances` to
    store the range to which category id is mapped.
    """

    def __init__(self, instances: "Instances", category_range: Tuple[int]):
        super(MappedInstances, self).__init__(instances.image_size, **instances._fields)
        self._category_range = category_range

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "MappedInstances":
        """
        Returns:
            MappedInstances: all fields are called with a `to(device)`,
            if the field has this method.
        """
        # CHANGE
        ret = MappedInstances(Instances(self._image_size), self._category_range)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "MappedInstances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        # CHANGE
        ret = MappedInstances(Instances(self._image_size), self._category_range)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    @staticmethod
    def transfer_meta_from_others(
        to_list: List[Instances], from_list: List[Instances]
    ) -> List[Instances]:
        """
        If `from_list` is a list of `MappedInstances`, pass the mask to `to_list`.
        Otherwise, keep `to_list` unchanged.
        """
        try:
            return [
                MappedInstances(to, fr._category_range)
                for to, fr in zip(to_list, from_list)
            ]
        except AttributeError:
            return to_list


@ROI_HEADS_REGISTRY.register()
class SoftmaxROIHeads(StandardROIHeads):
    """
    Used to be named "XRayROIHeads".

    It inherits StandardROIHeads, which is "standard" in some sense.
    The difference is adding a hack in `label_and_sample_proposals()` for
    the case when multiple gt boxes have same overlap with a target box
    when `MODEL.ROI_HEADS.JITTER_MATCH_QUALITY` is true.
    It is important for hierarchical labels.
    """

    @configurable
    def __init__(self, jitter_match_quality: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.jitter_match_quality = jitter_match_quality

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["jitter_match_quality"] = cfg.MODEL.ROI_HEADS.JITTER_MATCH_QUALITY
        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Same as
        `detectron2.modeling.roi_heads.roi_heads.ROIHeads.label_and_sample_proposals`,
        which is used in `StandardROIHeads`, except that a hack to jitter the values
        of `match_quality_matrix`.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # CHANGE: use `jitter_match_quality` to decide whether to hack
            if self.jitter_match_quality:
                # Below is a hack to jitter the values so that when multiple gt boxes
                # havesame overlap with a target box, matcher can choose one of them at
                # random every time. This is important for hierarchical labels, where
                # same box corresponds to a child and parent class (eg: dog, mammal)
                match_quality_matrix += (
                    1e-5 * torch.empty_like(match_quality_matrix).uniform_()
                )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return MappedInstances.transfer_meta_from_others(proposals_with_gt, targets)


@ROI_HEADS_REGISTRY.register()
class SigmoidROIHeads(SoftmaxROIHeads):
    """
    Used to be named "FocalLossROIHeads".

    It inherits XRayROIHeads.
    The differences is changing the `box_predictor` from
    `detectron2.modeling.roi_heads.roi_heads.FastRCNNOutputLayers` to
    `.xray_box_predictor.FocalLossOutputLayers`.
    The cross entropy loss in the box predictor is then replaced with focal loss,
    of which binary cross entropy loss is a special case.
    """

    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     ret = super().from_config(cfg, input_shape)
    #     box_head = ret["box_head"]
    #     ret["box_predictor"] = FocalLossOutputLayers(cfg, box_head.output_shape)
    #     return ret
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        # update output layer
        ret["box_predictor"] = FocalLossOutputLayers(cfg, ret["box_head"].output_shape)
        return ret


@ROI_HEADS_REGISTRY.register()
class BiStandardROIHeads(SigmoidROIHeads):
    """
    Replace the box_predictor: FewShotFocalLossOutputLayers
    FewShotFocalLossOutputLayers should handle both normal loss and detector loss (or proposals output)
    """

    @configurable
    def __init__(
        self,
        *,
        episodic_learning: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episodic_learning = episodic_learning

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["episodic_learning"] = cfg.MODEL.META_LEARN.EPISODIC_LEARNING
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        # update output layer
        ret["box_predictor"] = BiFocalLossOutputLayers(
            cfg, ret["box_head"].output_shape
        )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        class_codes: Optional[Dict[str, torch.Tensor]] = None,
        class_codes_target: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if not self.episodic_learning:  # calls FocalLossOutput
            return super().forward(images, features, proposals, targets)
        assert class_codes is not None
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            assert class_codes_target is not None
            losses = self._forward_box(
                features, proposals, class_codes, class_codes_target
            )
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, class_codes)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        class_codes: Optional[Dict[str, torch.Tensor]] = None,
        class_codes_target: Optional[List[torch.Tensor]] = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, class_codes)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(
                predictions=predictions,
                proposals=proposals,
                support_set_target=class_codes_target,
            )
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

import logging
import math
from typing import Optional

import torch
from detectron2.layers import cat
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class ExpScale(nn.Module):
    def __init__(self, init_value=0.0):
        super(ExpScale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * torch.exp(self.scale)

class CondConvBasic(nn.Module):
    def __init__(
        self,
        padding: int=0,
        stride: int = 1,
        use_bias: bool = True

    ):
        """
        Conditional Conv2d layer where both features and weight needs to be passed in.
        Args:
            padding: padding size in the kernel
            scale_score: once turn on, adds a scale to the final conv output
            l2_norm_weight: once turn on, l2 norm the weights before conv2d operation
            scale_value: set the init value of the scale
        """
        super(CondConvBasic, self).__init__()
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias

    def forward(self, feature: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        assert feature.size(1) == weight.size(1)  # intermediate channel, in, out
        assert feature.dim()== 4, f"Feature has dimension: {feature.dim()}"
        assert weight.dim()== 4, f"Weight has dimension: {weight.dim()}"
        if self.use_bias:
            # TODO: bias will be used to scale the weights
            cls_score = F.conv2d(
                feature,
                weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            cls_score = F.conv2d(
            feature,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        return cls_score

class CondConv(nn.Module):
    def __init__(
        self,
        padding: int=0,
        scale_score:bool =False,
        l2_norm_weight: bool = False,
        scale_value: float = 1.0,
    ):
        """
        Conditional Conv2d layer where both features and weight needs to be passed in.
        Args:
            padding: padding size in the kernel
            scale_score: once turn on, adds a scale to the final conv output
            l2_norm_weight: once turn on, l2 norm the weights before conv2d operation
            scale_value: set the init value of the scale
        """
        super(CondConv, self).__init__()
        self.padding = padding
        self.scale = Scale(init_value=scale_value) if scale_score else None
        self.l2_norm_weight = l2_norm_weight

    def forward(self, feature: torch.Tensor, weight: torch.Tensor, bias: bool=None):
        assert feature.size(1) == weight.size(1)  # intermediate channel, in, out
        assert feature.dim()== 4, f"Feature has dimension: {feature.dim()}"
        assert weight.dim()== 4, f"Weight has dimension: {weight.dim()}"
        # l2 normalize the feature
        if self.l2_norm_weight:
            weight = F.normalize(weight, p=2, dim=1)
        cls_score = F.conv2d(
            feature,
            weight,
            bias=bias,
            stride=1,
            padding=self.padding,
        )
        return self.scale(cls_score) if self.scale is not None else cls_score


class CondConvBlock(nn.Module):
    def __init__(self, padding=0, weight_len=256, scale_weight=False, use_scale=True):
        super(CondConvBlock, self).__init__()
        self.padding = padding
        self.weight_len = weight_len
        self.num_conv_layers = self.weight_len // 256
        self.condconvs = []
        self.use_scale = use_scale
        for _ in range(self.num_conv_layers):
            self.condconvs.append(CondConv(padding, scale_weight))
        if use_scale:
            self.scales = nn.ModuleList(
                [
                    Scale(init_value=1.0 / self.num_conv_layers)
                    for i in range(self.num_conv_layers)
                ]
            )
        self.condconvs = nn.ModuleList(self.condconvs)

    def forward(self, feature, weight: torch.Tensor, bias=None):
        assert len(weight.shape) == 4, f"weight has wrong shape, {weight.shape}"
        # bias is shared for different weights
        assert (
            weight.size(1) == 256 * self.num_conv_layers
        ), f"weight channel {weight.size(1)} is not {256 * self.num_conv_layers}"
        s, e = 0, 256
        if self.use_scale:
            outputs = self.scales[0](
                self.condconvs[0](feature, weight[:, s:e, :, :], bias)
            )
        else:
            outputs = self.condconvs[0](feature, weight[:, s:e, :, :], bias)
        for i in range(self.num_conv_layers - 1):
            s += 256
            e += 256
            if self.use_scale:
                outputs += self.scales[i](
                    self.condconvs[i](feature, weight[:, s:e, :, :], bias)
                )
            else:
                outputs += self.condconvs[i](feature, weight[:, s:e, :, :], bias)
        return outputs


class CosineSimilarityConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0):
        super(CosineSimilarityConv2d, self).__init__()
        self.cls_layer = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale = ExpScale(init_value=0.0)
        self.gn = nn.GroupNorm(32, in_channel)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_layer.bias, bias_value)
        # torch.nn.init.normal_(self.cls_layer.weight, std=0.01)

    def forward(self, x: torch.Tensor):
        # weight_norm = (
        #     torch.norm(self.cls_layer.weight.data, p=2, dim=1)
        #     .unsqueeze(1)
        #     .expand_as(self.cls_layer.weight.data)
        # )
        # self.cls_layer.weight.data = self.cls_layer.weight.data.div(weight_norm + 1e-12)
        # group normalize

        # normalize x
        self.cls_layer.weight.data = self.gn(self.cls_layer.weight.data) * torch.norm(self.cls_layer.weight.data, p=2, dim=1, keepdim=True)
        self.cls_layer.weight.data = torch.nn.functional.normalize(self.cls_layer.weight.data, p=2, dim=1)
        # x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        return self.scale(self.cls_layer(x))


class LinearModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        """
        Args:
            x: shape (batch_size, num_queries, C_in)
            weight: shape (C_out, C_in)
            bias: shape (C_out,)
        """
        return torch.nn.functional.linear(x, weight, bias)


def smooth_l1_loss_with_weight(
    input: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    reduction: str = "none",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weight is None:
        return smooth_l1_loss(input, target, beta, reduction)

    loss = smooth_l1_loss(input, target, beta) * weight
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def sigmoid_focal_loss_with_mask(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Similar to `fvcore.nn.sigmoid_focal_loss`, except allowing a mask.
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        mask: (optional) A mask to show which logits are calculated in loss
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, weight=mask, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def extract_mask(proposals, num_classes):
    """
    If it is a list of `MaskedInstance`, extract the mask information.

    Args:
        proposals (list[Instances])

    Returns:
        mask (Optional(torch.Tensor)): if it returns, the shape is N x K,
            where N is the overall proposal number for all images,
                  K is the overall class numbers for all datasets.
    """
    if len(proposals):
        masks = []
        try:
            for proposals_per_image in proposals:
                start_class_id, end_class_id = proposals_per_image._category_range
                mask_this_image = cat(
                    [
                        torch.zeros(len(proposals_per_image), start_class_id),
                        torch.ones(
                            len(proposals_per_image), end_class_id - start_class_id
                        ),
                        torch.zeros(
                            len(proposals_per_image), num_classes - end_class_id
                        ),
                    ],
                    1,
                )
                masks.append(mask_this_image)
            return cat(masks, 0)
        except AttributeError:
            return torch.ones(sum(len(p) for p in proposals), num_classes)
    else:
        return torch.ones(0, num_classes)

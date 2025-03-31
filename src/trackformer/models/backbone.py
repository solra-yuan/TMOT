# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .multi_input_intermediate_layer_getter import MultiInputIntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelMaxPool)

from dataclasses import dataclass

from ..util.misc import NestedTensor, is_main_process
from abc import ABC, abstractmethod


@dataclass
class BackboneOptions:
    name: str
    train_backbone: bool
    return_interm_layers: bool
    dilation: bool


@dataclass
class BackboneProperties(ABC):
    strides: list
    num_channels: list
    layer_names: set
    return_layers: dict

    @abstractmethod
    def __init__(self, options: BackboneOptions):
        pass


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        properties: BackboneProperties,
        options: BackboneOptions
    ):
        super().__init__()

        self.strides = properties.strides
        self.num_channels = properties.num_channels

        for name, parameter in backbone.named_parameters():
            if (not options.train_backbone or all(layer_name not in name for layer_name in properties.layer_names)):
                parameter.requires_grad_(False)

        # gamma
        self.body = MultiInputIntermediateLayerGetter(
            backbone,
            return_layers=properties.return_layers
        )

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        model: nn.Module,
        properties: BackboneProperties,
        options: BackboneOptions,
    ):
        replace_stride_with_dilation = [False, False, options.dilation]
        pretrained = is_main_process()
        norm_layer = FrozenBatchNorm2d

        backbone = model(
            replace_stride_with_dilation=replace_stride_with_dilation,
            pretrained=pretrained,
            norm_layer=norm_layer
        )

        super().__init__(
            backbone,
            properties,
            options,
        )

        if options.dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

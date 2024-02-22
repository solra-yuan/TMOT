import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, conv1x1

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'resnet50_preprocessing_type_a']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet4Channel(ResNet):
    def __init__(
            self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer
        )

        self.conv_rgbt_to_latent = nn.Conv2d(
            4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.rgbt_bn = nn.BatchNorm2d(self.inplanes)
        self.rgbt_relu = nn.ReLU(inplace=True)
        self.rgbt_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        rgbt_downsample = nn.Sequential(
            conv1x1(self.inplanes, 256, stride=1),
            norm_layer(256)
        )
        self.preprocess_latent_channel = nn.Sequential(
            Bottleneck(self.inplanes, 64,
                       stride=1,
                       downsample=rgbt_downsample,
                       groups=1,
                       base_width=64,
                       dilation=1)
        )
        self.conv_inplane_to_rgb = nn.Conv2d(
            self.inplanes*4, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_rgbt_to_latent(x)
        x = self.rgbt_bn(x)
        x = self.rgbt_relu(x)
        x = self.rgbt_maxpool(x)
        x = self.preprocess_latent_channel(x)
        x = self.conv_inplane_to_rgb(x)

        return super().forward(x)


def _resnet_4_channel(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet4Channel:
    model = ResNet4Channel(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50_4_channel(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_resnet = _resnet_4_channel('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                                          **kwargs)

    return pretrained_resnet

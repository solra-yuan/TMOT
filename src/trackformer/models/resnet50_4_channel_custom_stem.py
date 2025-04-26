from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.utils import _log_api_usage_once

from .backbone import BackboneProperties, BackboneOptions

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'resnet50_4_channel']

# 모델 URL 정의
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

@dataclass
class ResNet4ChannelCustomStemProperties(BackboneProperties):
    def __init__(self, options: BackboneOptions):
        self.layer_names = set([
            'layer2',
            'layer3',   
            'layer4',
            'conv_rgbt_to_latent',  # alpha-beta
            'rgbt_bn',
            'conv_inplane_to_rgb',
            'process_rgb',  # gamma
            'process_t',
            'process_rgb2',
            'process_t2',
        ])

        if options.return_interm_layers:
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
            self.return_layers = {
                "conv_rgbt_to_latent": "0",
                "rgbt_maxpool": "1",
                "conv_inplane_to_rgb": "2",
                "layer1": "3",
                "layer2": "4",
                "layer3": "5",
                "layer4": "6"
            }
        else:
            self.strides = [32]
            self.num_channels = [2048]
            self.return_layers = {
                'layer4': "1"
            }


class MultiKernelConvBlock(nn.Module):
    """
    4채널 입력을 받아 아래 순서로 convolution을 수행:
      - (11*11): 4 → 8, stride=stride, padding=5 → downsampling (1/2)
      - (9*9):   8 → 16, stride=2, padding=4 → 추가 downsampling (1/2)
      - (7*7):  16 → 32, stride=1, padding=3
      - (5*5):  32 → 64, stride=1, padding=2
      - (3*3):  64 → 128, stride=1, padding=1
    최종 출력은 (batch, 128, H/4, W/4) 형태이며,
    이후 conv_inplane_to_rgb에서 1*1 convolution을 통해 128채널 → 64채널로 변환합니다.
    """

    def __init__(self, in_channels: int, stride: int = 2):
        super(MultiKernelConvBlock, self).__init__()
        self.conv11 = nn.Conv2d(
            in_channels, 8, kernel_size=11, stride=stride, padding=5, bias=False)
        self.bn11 = nn.BatchNorm2d(8)

        # conv9: stride=2로 추가 downsampling
        self.conv9 = nn.Conv2d(8, 16, kernel_size=9,
                               stride=2, padding=4, bias=False)
        self.bn9 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(16, 32, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class ResNet4ChannelCustomStem(nn.Module):
    """
    4채널(RGBT) 입력을 처리하기 위해 ResNet을 확장한 클래스.

    변경 사항:
      - 기존 7*7 conv 대신, 수정된 multi-kernel conv block을 사용하여 4채널 입력을 latent feature로 변환.
        사용되는 커널 크기 및 채널 변화:
            (11*11): 4 → 8   (stride=stride로 downsampling, 예: 2)
            (9*9):    8 → 16  (stride=2로 추가 downsampling)
            (7*7):   16 → 32
            (5*5):   32 → 64
            (3*3):   64 → 128
      - 이후 conv_inplane_to_rgb에서 1*1 conv를 사용하여 128채널을 64채널로 정합,
        ResNet을 layer이 기대하는 64채널 입력을 제공합니다.
    """

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
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation은 None 또는 "
                f"3개의 요소를 포함한 리스트여야 합니다: {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 4채널 입력을 수정된 multi-kernel conv block을 통해 latent feature로 변환
        self.conv_rgbt_to_latent = MultiKernelConvBlock(4, stride=2)
        # conv_rgbt_to_latent에서 conv11과 conv9의 downsampling으로 전체 해상도가 1/4 축소됨
        self.rgbt_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv_inplane_to_rgb에서 1*1 conv를 통해 128채널을 64채널로 변환
        self.conv_inplane_to_rgb = nn.Conv2d(
            128, 64, kernel_size=1, stride=1, bias=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        잔차 레이어를 생성합니다.

        매개변수:
            block (Type[Union[BasicBlock, Bottleneck]]): 블록 유형.
            planes (int): 레이어의 출력 채널 수.
            blocks (int): 레이어 내 블록 수.
            stride (int): 첫 번째 블록의 stride 값.
            dilate (bool): 합성곱에서 dilation을 사용할지 여부.

        반환값:
            nn.Sequential: 잔차 블록들의 연속적 컨테이너.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def preprocessing_layer(self, x):
        # x: (batch, 4, H, W) - RGBT 입력
        # 1. 수정된 multi-kernel conv block을 통해 latent feature 추출 (출력: (batch, 128, H/4, W/4))
        x = self.conv_rgbt_to_latent(x)
        # 추가 downsampling → 해상도 축소 (예: (batch, 128, H/8, W/8))
        x = self.rgbt_maxpool(x)

        # 3. conv_inplane_to_rgb에서 1*1 conv로 128채널을 64채널로 정합
        x = self.conv_inplane_to_rgb(x)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        내부 순전파 로직을 정의합니다.

        매개변수:
            x (Tensor): 입력 텐서, 형태 (N, C, H, W).

        반환값:
            Tensor: 네트워크를 통과한 출력 텐서.
        """

        x = self.preprocessing_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        네트워크의 순전파를 실행합니다.

        매개변수:
            x (Tensor): 입력 텐서, 형태 (N, C, H, W).

        반환값:
            Tensor: 네트워크를 통과한 출력 텐서.
        """
        return self._forward_impl(x)


def _resnet_4_channel_custom_stem(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet4ChannelCustomStem:
    """
    4채널 입력을 처리할 수 있는 ResNet 모델 생성.

    Args:
        arch (str): 모델 아키텍처 이름.
        block (Type[Union[BasicBlock, Bottleneck]]): ResNet 블록 유형.
        layers (List[int]): 각 레이어의 블록 수.
        pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
        progress (bool): 다운로드 진행 상황을 출력할지 여부.

    Returns:
        ResNet4ChannelCustomStem: 4채널 입력을 처리할 수 있는 ResNet 모델.
    """
    model = ResNet4ChannelCustomStem(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50_4_channel_custom_stem(
    pretrained: bool = False,
    progress: bool = True, 
    **kwargs: Any
) -> ResNet4ChannelCustomStem:
    """
    ResNet-50 아키텍처를 기반으로 한 4채널 입력 모델 생성.
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
        progress (bool): 다운로드 진행 상황을 출력할지 여부.

    Returns:
        ResNet4Channel: 4채널 입력을 처리할 수 있는 ResNet-50 모델.
    """
    pretrained_resnet = _resnet_4_channel_custom_stem(
        'resnet50',
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )

    return pretrained_resnet
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from .CustomResNet import CustomResNet

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

class ResNet4Channel(CustomResNet):
    """
    4채널(RGBT) 입력을 처리할 수 있도록 ResNet을 확장한 클래스.

    CustomResNet을 상속받아 4채널 입력 데이터를 처리하기 위해 전처리 로직과 네트워크를 정의했습니다.

    주요 기능:
        - RGBT 데이터를 처리하기 위한 초기 합성곱 레이어 추가.
        - 입력 데이터를 3채널(RGB)로 변환하는 추가 레이어 정의.
        - 기존 ResNet 모델 아키텍처와 호환 가능.

    Attributes:
        conv_rgbt_to_latent (nn.Conv2d): RGBT 데이터를 latent 공간으로 변환하는 합성곱 레이어.
        rgbt_bn (nn.BatchNorm2d): RGBT 데이터 정규화를 위한 BatchNorm 레이어.
        rgbt_relu (nn.ReLU): 활성화 함수.
        rgbt_maxpool (nn.MaxPool2d): RGBT 데이터에 적용할 풀링 레이어.
        preprocess_latent_channel (nn.Sequential): Bottleneck 블록을 활용한 전처리 계층.
        conv_inplane_to_rgb (nn.Conv2d): 4채널 데이터를 3채널로 변환하는 레이어.
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
        """
        ResNet4Channel 클래스 초기화.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): ResNet 블록 유형.
            layers (List[int]): 각 레이어의 블록 수.
            num_classes (int): 분류 클래스 수.
            zero_init_residual (bool): 잔차 분기의 BatchNorm 레이어를 0으로 초기화할지 여부.
            groups (int): 그룹화된 합성곱의 그룹 수.
            width_per_group (int): 그룹당 너비.
            replace_stride_with_dilation (Optional[List[bool]]): stride를 dilation으로 대체할지 여부.
            norm_layer (Optional[Callable[..., nn.Module]]): 정규화 레이어.
        """
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

    def preprocessing_sequence(self):
        """
        4채널 입력 데이터를 처리하기 위한 전처리 단계 정의.

        - RGBT 데이터를 latent 공간으로 변환.
        - Bottleneck 블록을 통해 전처리.
        - 4채널 데이터를 3채널(RGB)로 변환.
        """
        self.conv_rgbt_to_latent = nn.Conv2d(
            4,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.rgbt_bn = nn.BatchNorm2d(self.inplanes)
        self.rgbt_relu = nn.ReLU(inplace=True)
        self.rgbt_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        rgbt_downsample = nn.Sequential(
            conv1x1(self.inplanes, 256, stride=1),
            self._norm_layer(256)
        )
        self.preprocess_latent_channel = nn.Sequential(
            Bottleneck(
                self.inplanes, 64,
                stride=1,
                downsample=rgbt_downsample,
                groups=1,
                base_width=64,
                dilation=1
            )
        )

        self.conv_inplane_to_rgb = nn.Conv2d(
            self.inplanes * 4, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        네트워크의 순전파 로직.

        Args:
            x (Tensor): 입력 텐서, 형태 (N, 4, H, W).

        Returns:
            Tensor: 네트워크를 통과한 출력 텐서.
        """
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
    """
    4채널 입력을 처리할 수 있는 ResNet 모델 생성.

    Args:
        arch (str): 모델 아키텍처 이름.
        block (Type[Union[BasicBlock, Bottleneck]]): ResNet 블록 유형.
        layers (List[int]): 각 레이어의 블록 수.
        pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
        progress (bool): 다운로드 진행 상황을 출력할지 여부.

    Returns:
        ResNet4Channel: 4채널 입력을 처리할 수 있는 ResNet 모델.
    """
    model = ResNet4Channel(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet50_4_channel(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet4Channel:
    """
    ResNet-50 아키텍처를 기반으로 한 4채널 입력 모델 생성.

    Args:
        pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
        progress (bool): 다운로드 진행 상황을 출력할지 여부.

    Returns:
        ResNet4Channel: 4채널 입력을 처리할 수 있는 ResNet-50 모델.
    """
    pretrained_resnet = _resnet_4_channel(
        'resnet50',
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )

    return pretrained_resnet

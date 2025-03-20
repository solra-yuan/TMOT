from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from abc import abstractmethod

class CustomResNet(nn.Module):
    """
    사용자 정의 ResNet 모델.

    이 클래스는 ResNet의 기본 구조를 기반으로 하며, `preprocessing_sequence` 메서드를 통해 데이터 전처리 로직을
    사용자 정의할 수 있도록 설계되었습니다. ResNet의 기본 블록은 BasicBlock과 Bottleneck을 지원하며,
    torchvision 스타일의 ResNet API와 호환됩니다.

    속성:
        inplanes (int): 초기 입력 채널 수.
        dilation (int): 현재 dilation 값 (팽창 합성곱에 사용됨).
        groups (int): 그룹화된 합성곱의 그룹 수.
        base_width (int): 그룹당 합성곱의 너비.
        conv1 (nn.Conv2d): 초기 합성곱 레이어.
        bn1 (nn.Module): 초기 합성곱 뒤의 배치 정규화 레이어.
        relu (nn.ReLU): ReLU 활성화 함수.
        maxpool (nn.MaxPool2d): 초기 합성곱 뒤의 최대 풀링 레이어.
        layer1-4 (nn.Sequential): 특징 추출을 위한 잔차 레이어.
        avgpool (nn.AdaptiveAvgPool2d): 전역 평균 풀링 레이어.
        fc (nn.Linear): 최종 분류를 위한 완전 연결 레이어.

    메서드:
        __init__: ResNet 아키텍처 초기화.
        _make_layer: 지정된 매개변수로 잔차 레이어 생성.
        _forward_impl: 순전파 로직 정의.
        forward: 순전파를 실행하는 공개 메서드.
        preprocessing_sequence: 하위 클래스에서 구현해야 하는 추상 메서드로, 사용자 정의 전처리 로직을 정의.

    주의:
        `preprocessing_sequence` 메서드는 초기화 단계에서 가장 먼저 호출되도록 설계되었습니다.
        이는 PyTorch의 `nn.Module`이 코드에 정의된 순서대로 레이어를 처리하기 때문입니다.
        전처리 로직을 최상단에 배치함으로써 모든 입력 데이터가 네트워크 레이어에 전달되기 전에
        전처리를 거치도록 보장합니다. 이를 통해 모델 구조의 일관성과 데이터 흐름의 명확성을 유지할 수 있습니다.

        또한, `nn.Module`은 속성을 정의한 순서대로 모듈을 등록하고 `state_dict` 및 `forward` 호출에서 이 순서를 따릅니다.
        따라서 전처리 단계가 올바르게 실행되기 위해서는 이를 초기화 단계의 가장 위에 배치하는 것이 중요합니다.
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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        ResNet 아키텍처를 사용자 정의 설정으로 초기화합니다.

        매개변수:
            block (Type[Union[BasicBlock, Bottleneck]]): 블록 유형 (BasicBlock 또는 Bottleneck).
            layers (List[int]): 각 레이어의 블록 수.
            num_classes (int): 최종 분류 레이어의 출력 클래스 수.
            zero_init_residual (bool): 잔차 분기의 마지막 BatchNorm 레이어를 0으로 초기화할지 여부.
            groups (int): 그룹화된 합성곱의 그룹 수.
            width_per_group (int): 그룹당 너비.
            replace_stride_with_dilation (Optional[List[bool]]): stride를 dilation으로 대체하는 설정.
            norm_layer (Optional[Callable[..., nn.Module]]): 사용할 정규화 레이어 (기본값: BatchNorm2d).

        주의:
            `preprocessing_sequence` 메서드는 초기화의 시작 부분에서 호출되어 전처리 로직을 정의합니다.
            이 설계는 PyTorch의 순차적인 모듈 실행 순서에서 올바른 연산 순서를 유지하기 위해 필수적입니다.
            또한, `nn.Module`의 동작 특성상 속성을 정의한 순서대로 레이어가 처리되고 등록되므로,
            전처리 단계가 다른 레이어보다 먼저 실행되도록 보장하려면 초기화의 가장 위에 위치해야 합니다.
        """
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

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

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        내부 순전파 로직을 정의합니다.

        매개변수:
            x (Tensor): 입력 텐서, 형태 (N, C, H, W).

        반환값:
            Tensor: 네트워크를 통과한 출력 텐서.
        """

        x = self.preprocessing_sequence(x)

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

    def preprocessing_sequence(self):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

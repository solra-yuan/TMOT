import torch


class NormalizeHelper:
    """
    이미지 텐서를 정규화하기 위한 유틸리티 클래스입니다.

    이 클래스는 두 가지 정규화 방법을 제공합니다.
    1. per_channel: 각 채널별로 독립적으로 정규화.
    2. across_channels: 모든 채널에 대해 정규화.

    정규화는 입력 텐서를 특정 범위로 스케일링하여, 딥러닝 모델 학습 과정에서 데이터의 편향을 줄이고
    성능을 향상시키는 데 도움을 줍니다.

    사용 예:
        >>> image = torch.rand(3, 256, 256)  # [C, H, W] 형태의 랜덤 이미지 생성
        >>> normalized_image = NormalizeHelper.per_channel(image)
        >>> normalized_across = NormalizeHelper.across_channels(image)
    """

    @staticmethod
    def per_channel(image: torch.Tensor):
        """
        입력 텐서의 각 채널을 독립적으로 정규화합니다.

        Args:
            image (torch.Tensor): 정규화할 이미지 텐서. (3, H, W) 형태를 가져야 합니다.

        Returns:
            torch.Tensor: 각 채널이 독립적으로 정규화된 텐서.

        Raises:
            TypeError: 입력 값이 torch.Tensor가 아닌 경우.
            ValueError: 입력 텐서가 (3, H, W) 형태가 아닌 경우.
        """

        # 입력 값이 torch.Tensor인지 확인
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor")

        # 입력 텐서의 형태가 (3, H, W)인지 확인
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Input tensor must have shape (3, H, W).")

        # 각 채널별 최소값 계산: [C, H, W] -> [C, 1, 1]
        min_val = image.min(dim=-1, keepdim=True)[0] \
                       .min(dim=-2, keepdim=True)[0]

        # 각 채널별 최대값 계산: [C, H, W] -> [C, 1, 1]
        max_val = image.max(dim=-1, keepdim=True)[0] \
                       .max(dim=-2, keepdim=True)[0]

        # 최소값과 최대값 사이의 범위 계산
        range_val = max_val - min_val

        # 0으로 나누기 방지를 위해 range_val에 작은 값을 추가
        range_val += (range_val == 0).float()

        # 각 채널을 (값 - 최소값) / (최대값 - 최소값) 형태로 정규화
        normalized_image = (image - min_val) / range_val
        return normalized_image

    @staticmethod
    def across_channels(image: torch.Tensor):
        """
        입력 텐서를 모든 채널에 대해 정규화합니다.

        Args:
            image (torch.Tensor): 정규화할 이미지 텐서. (C, H, W) 형태를 가져야 합니다.

        Returns:
            torch.Tensor: 모든 채널에 대해 정규화된 텐서.
        """

        # 텐서의 전체 최소값 계산
        min_val = image.min()

        # 텐서의 전체 최대값 계산
        max_val = image.max()

        # 모든 채널에 대해 정규화: (값 - 최소값) / (최대값 - 최소값)
        normalized_image = (image - min_val) / (max_val - min_val)

        return normalized_image

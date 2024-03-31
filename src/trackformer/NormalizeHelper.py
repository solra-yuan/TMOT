import torch


class NormalizeHelper:
    @staticmethod
    def per_channel(image):
        """Normalize each channel of the image tensor independently."""

        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor")

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Input tensor must have shape (3, H, W).")

        # image shape: [C, H, W]
        min_val = image.min(dim=-1, keepdim=True)[0] \
                       .min(dim=-2, keepdim=True)[0]

        max_val = image.max(dim=-1, keepdim=True)[0] \
                       .max(dim=-2, keepdim=True)[0]

        # 최소값과 최대값 사이의 범위 계산
        range_val = max_val - min_val

        # 0으로 나누기 방지를 위해 range_val에 작은 값을 추가
        range_val += (range_val == 0).float()

        # 정규화: 채널별로 (값 - 최소값) / (최대값 - 최소값)
        normalized_image = (image - min_val) / range_val
        return normalized_image

    @staticmethod
    def across_channels(image):
        """Normalize the image tensor across all channels."""
        # image shape: [C, H, W]
        min_val = image.min()
        max_val = image.max()

        # Normalize across all channels
        normalized_image = (image - min_val) / (max_val - min_val)

        return normalized_image

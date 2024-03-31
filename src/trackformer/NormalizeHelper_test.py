import torch
from .NormalizeHelper import NormalizeHelper
import pytest


def test_per_channel():
    # Create a test tensor with shape [C, H, W]
    image = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])

    # Normalize per channel
    normalized_image = NormalizeHelper.per_channel(image)

    # Check if the result is as expected
    expected_min = 0
    expected_max = 1
    assert normalized_image.min().item() == expected_min, "Minimum value should be 0."
    assert normalized_image.max().item() == expected_max, "Maximum value should be 1."

if __name__ == "__main__":
    pytest.main()

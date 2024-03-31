import torch
from .NormalizeHelper import NormalizeHelper
import pytest


@pytest.mark.parametrize('image', [(
    torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 4.0], [6.0, 8.0]],
        [[0.0, 1.0], [1.0, 0.0]]
    ])
), (
    torch.tensor([
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0]],
        [[2.0, 4.0, 6.0], [6.0, 8.0, 10.0], [0.0, 1.0, -1.0]],
        [[0.0, 1.0, -1.0], [-1.0, 1.0, 0.0], [3.0, 4.0, 5.0]],
    ])
)])
def test_per_channel(image):
    # Normalize per channel
    normalized_image = NormalizeHelper.per_channel(image)

    for c in range(normalized_image.size(0)):
        channel = normalized_image[c]
        assert torch.isclose(
            channel.min(),
            torch.tensor(0.0),
            atol=1e-5
        ), f"Channel {c} minimum value should be 0."
        assert torch.isclose(
            channel.max(),
            torch.tensor(1.0),
            atol=1e-5
        ), f"Channel {c} maximum value should be 1."


@pytest.mark.parametrize('image', [(
    torch.zeros(3, 10, 10)
)])
def test_per_channel_same(image):
    # Normalize per channel
    normalized_image = NormalizeHelper.per_channel(image)

    for c in range(normalized_image.size(0)):
        channel = normalized_image[c]
        assert torch.isclose(
            channel.min(),
            torch.tensor(0.0),
            atol=1e-5
        ), f"Channel {c} minimum value should be 0."


def test_per_channel_with_none():
    with pytest.raises(TypeError):
        NormalizeHelper.per_channel(None)


def test_per_channel_with_empty():
    with pytest.raises(ValueError):
        NormalizeHelper.per_channel(torch.empty(0))


def test_per_channel_with_same():
    # Create a test tensor with shape [C, H, W]
    image = torch.zeros(3, 10, 10)

    # Normalize per channel
    normalized_image = NormalizeHelper.per_channel(image)

    # Check if the result is as expected
    expected_min = 0
    expected_max = 0
    assert normalized_image.min().item() == expected_min, "Minimum value should be 0."
    assert normalized_image.max().item() == expected_max, "Maximum value should be 1."


def test_across_channels():
    # Create a test tensor with shape [C, H, W]
    image = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])

    # Normalize per channel
    normalized_image = NormalizeHelper.across_channels(image)

    # Check if the result is as expected
    expected_min = 0
    expected_max = 1
    assert normalized_image.min().item() == expected_min, "Minimum value should be 0."
    assert normalized_image.max().item() == expected_max, "Maximum value should be 1."


if __name__ == "__main__":
    pytest.main()

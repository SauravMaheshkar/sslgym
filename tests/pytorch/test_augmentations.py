"""Test custom pytorch augmentations."""

import pytest
from PIL import Image

from src.pytorch.augmentations import GaussianBlur, Solarization


@pytest.mark.pytorch
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_gaussian_blur(prob: float) -> None:
    """Test GaussianBlur."""
    image = Image.open("tests/Grumpy_Cat.jpeg")
    output = GaussianBlur(prob)(image)
    assert output.mode == image.mode
    assert output.size == image.size


@pytest.mark.pytorch
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_solarization(prob: float) -> None:
    """Test Solarization."""
    image = Image.open("tests/Grumpy_Cat.jpeg")
    output = Solarization(prob)(image)
    assert output.mode == image.mode
    assert output.size == image.size

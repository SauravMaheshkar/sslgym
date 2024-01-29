"""Test VICReg model."""

import pytest
import torch

from src.pytorch.models.vicreg import VICReg

BATCH_SIZE = 64


@pytest.mark.pytorch
@pytest.mark.pytorch_model
def test_vicreg():
    """Test VICReg model."""
    model = VICReg(arch="resnet18", pretrained=False, batch_size=BATCH_SIZE)
    view_1 = torch.rand(BATCH_SIZE, 3, 224, 224)
    view_2 = torch.rand(BATCH_SIZE, 3, 224, 224)
    loss = model(view_1, view_2)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

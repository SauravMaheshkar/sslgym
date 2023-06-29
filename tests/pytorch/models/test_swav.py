"""Test SwAV model."""
import pytest
import torch

from src.pytorch.models.swav import SwAV


BATCH_SIZE = 64


@pytest.mark.pytorch
@pytest.mark.pytorch_model
def test_vicreg():
    """Test VICReg model."""
    model = SwAV(normalize=True, hidden_mlp=2048, output_dim=128, nmb_prototypes=3000)
    view = torch.rand(BATCH_SIZE, 3, 224, 224)
    loss = model(view)
    assert isinstance(loss[0], torch.Tensor)
    assert loss[0].shape == torch.Size([BATCH_SIZE, 128])

"""
Implementation of VICReg model.

Based on Variance-Invariance-Covariance Regularization For Self-Supervised Learning

Authors: Adrien Bardes, Jean Ponce and Yann LeCun
"""

import typing
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as torchvision_models


class VICReg(nn.Module):
    """VICReg model class"""

    def __init__(
        self,
        batch_size: int,
        arch: str = "resnet50",
        pretrained: bool = True,
        mlp: str = "8192-8192-8192",
        sim_coeff: Optional[float] = 25.0,
        std_coeff: Optional[float] = 25.0,
        cov_coeff: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = int(mlp.split("-")[-1])
        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.encoder = torchvision_models.__dict__[arch](
            pretrained=pretrained, zero_init_residual=True
        )
        self.expander = expander(
            embedding=int(typing.cast(Tuple, self.encoder.fc.weight.shape)[1])
        )
        # This step is important as the implementation
        # of torchvision and the one used by the authors
        # are different
        self.encoder.fc = nn.Identity()

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the model and return the net loss.
        """
        # Get Embeddings
        view_1 = self.expander(self.encoder(view_1))
        view_2 = self.expander(self.encoder(view_2))

        # Calculate the Representation (Invariance) Loss
        repr_loss = F.mse_loss(view_1, view_2)

        # Calculate var. and std. dev. of embeddings
        view_1 = view_1 - view_1.mean(dim=0)
        view_2 = view_2 - view_2.mean(dim=0)
        std_x = torch.sqrt(view_1.var(dim=0) + 0.0001)
        std_y = torch.sqrt(view_2.var(dim=0) + 0.0001)

        # Calculate the Variance Loss (Hinge Function)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Get Covariance Matrix
        cov_x = (view_1.T @ view_1) / (self.batch_size - 1)
        cov_y = (view_2.T @ view_2) / (self.batch_size - 1)

        # Calculate the Covariance Loss
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # Weighted Avg. of Invariance, Variance and Covariance Loss
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


def expander(embedding: int) -> nn.Module:
    """
    Custom expander module to map the encoded representations
    into embeddings.

    * Representations -> view_2 = f(view_1)
    * Embeddings -> z = h(view_2)
    """
    mlp_spec = f"{embedding}-8192-8192-8192"
    layers: List[nn.Module] = []
    dims = list(map(int, mlp_spec.split("-")))
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the off-diagonal elements of a square matrix.
    """
    n, m = tensor.shape
    assert n == m, "Not a square tensor"
    return tensor.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

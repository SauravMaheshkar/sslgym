"""Popular Optimizer implementations."""

from typing import Callable, Optional

import torch


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS)

    * uses a separate learning rate for each layer and not for each weight.
    * the magnitude of the update is controlled w.r.t the weight norm.
    """

    def __init__(
        self,
        params,
        lr: Optional[float] = 0.01,
        weight_decay: Optional[float] = 1e-6,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: Optional[Callable] = None,
        lars_adaptation_filter: Optional[Callable] = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                param_grad = param.grad

                if param_grad is None:
                    continue

                if group["weight_decay_filter"] is None or not group[
                    "weight_decay_filter"
                ](param):
                    param_grad = param_grad.add(param, alpha=group["weight_decay"])

                if group["lars_adaptation_filter"] is None or not group[
                    "lars_adaptation_filter"
                ](param):
                    param_norm = torch.norm(param)
                    update_norm = torch.norm(param_grad)
                    one = torch.ones_like(param_norm)
                    q = torch.where(  # pylint: disable=invalid-name
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (group["eta"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    param_grad = param_grad.mul(q)

                param_state = self.state[param]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(param)
                mu = param_state["mu"]  # pylint: disable=invalid-name
                mu.mul_(group["momentum"]).add_(param_grad)

                param.add_(mu, alpha=-group["lr"])

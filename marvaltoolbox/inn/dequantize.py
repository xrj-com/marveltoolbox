import torch
import torch.nn as nn
import torch.nn.functional as F

class DequantizeLayer(nn.Module):

    def __init__(self, alpha: float = 1e-6, is_deq: bool = True, is_logit: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.is_deq = is_deq
        self.is_logit = is_logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deq:
            x = self._dequantize(x)
        if self.is_logit:
            x = self._logit_transform(x)
        return x

    def _dequantize(self, x):
        """
        Adds noise to pixels to dequantize them.
        """
        return x + torch.rand(*x.shape, device=x.device) / 256.0

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = self.alpha + (2 - self.alpha) * x
        return self._logit(x)

    def _logit(self, x, eps=1e-5):
        x.clamp_(eps, 1 - eps)
        return x.log() - (1 - x).log()

    def inverse(self, x):
        return F.sigmoid(x)
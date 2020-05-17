import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:
        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return normalized_input
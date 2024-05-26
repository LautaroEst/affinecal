

from typing import Literal
import torch
from torch import nn

SUPPORTED_SCALINGS = Literal["matrix", "vector", "scalar", "none"]


class AffineCalibrator(nn.Module):
    """
    Affine calibration block. It is a linear block that performs an affine transformation
    of the input feature vector ino order to output the calibrated logits.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    alpha : {"vector", "scalar", "matrix", "none"}, optional
        Type of affine transformation, by default "vector"
    beta : bool, optional
        Whether to use a beta term, by default True
    """    
    def __init__(
        self, 
        num_classes: int, 
        alpha: SUPPORTED_SCALINGS = "matrix",
        beta: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.additional_arguments = {
            "alpha": alpha,
            "beta": beta,
        }

        # Set the alpha parameter
        if alpha == "matrix":
            self.alpha = nn.Parameter(torch.zeros(num_classes, num_classes), requires_grad=True)
        elif alpha == "vector":
            self.alpha = nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        elif alpha == "scalar":
            self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)
        elif alpha == "none":
            self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=False)
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        # Set the beta parameter
        self.beta = nn.Parameter(torch.zeros(num_classes), requires_grad=beta)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.alpha.shape == (self.num_classes, self.num_classes):
            output = logits @ self.alpha.T
        elif self.alpha.shape in [(self.num_classes,), ()]:
            output = logits * self.alpha
        output = output + self.beta

        return output
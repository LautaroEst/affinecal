

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

        # Set the alpha parameter
        if alpha == "matrix":
            self.alpha = nn.Parameter(torch.eye(num_classes, num_classes), requires_grad=True)
            self._forward = self._forward_matrix_of_vector
        elif alpha == "vector":
            self.alpha = nn.Parameter(torch.ones(num_classes, 1), requires_grad=True)
            self._forward = self._forward_vector
        elif alpha == "scalar":
            self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)
            self._forward = self._forward_scalar
        elif alpha == "none":
            self.alpha = torch.tensor(1.)
            self._forward = self._forward_scalar
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        # Set the beta parameter
        if beta:
            self.beta = nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        else:
            self.beta = torch.zeros(num_classes)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self._forward(logits)
    
    def _forward_matrix_of_vector(self, logits: torch.Tensor) -> torch.Tensor:
        return logits @ self.alpha.T + self.beta
    
    def _forward_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.alpha + self.beta
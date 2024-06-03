
from typing import Literal
from torch import nn

SUPPORTED_LOSSES = Literal["cross_entropy", "brier"]

def init_loss(loss):
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss == "brier":
        return nn.BCELoss()
    else:
        raise ValueError(f"Invalid loss: {loss}")


from typing import Literal, Optional

import torch

from affinecal import (
    AffineCalibrator, 
    Trainer,
    SUPPORTED_SCALINGS,
    SUPPORTED_LOSSES
)



def main(
    train_logits: torch.Tensor,
    train_labels: torch.Tensor,
    val_logits: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    output_dir: Optional[str] = "./output",
    alpha: Optional[SUPPORTED_SCALINGS] = "vector",
    beta: Optional[bool] = True,
    accelerator: Optional[Literal["cpu", "cuda"]] = "cpu",
    learning_rate: Optional[float] = 0.01,
    max_ls: Optional[int] = 100,
    epochs: Optional[int] = 100,
    loss: Optional[SUPPORTED_LOSSES] = "cross_entropy",
):
    
    # Initialize the model
    num_classes = train_logits.shape[1]
    calibrator = AffineCalibrator(
        num_classes = num_classes,
        alpha = alpha,
        beta = beta,
    )

    # Initialize the trainer
    trainer = Trainer(
        accelerator = accelerator,
        learning_rate = learning_rate,
        max_ls = max_ls,
        epochs = epochs,
        loss = loss,
        output_dir = output_dir,
    )

    trainer.fit(calibrator, train_logits, train_labels, val_logits, val_labels)

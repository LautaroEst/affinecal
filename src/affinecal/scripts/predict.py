

import os
from typing import Optional
import torch

from affinecal import AffineCalibrator


def main(
    logits: torch.Tensor,
    model_path: str,
    output_dir: Optional[str] = "./output",
):

    model_state = torch.load(model_path)    
    calibrator = AffineCalibrator(
        num_classes=model_state["num_classes"],
        alpha=model_state["alpha"],
        beta=model_state["beta"],
    )
    calibrator.load_state_dict(model_state["state_dict"])
    calibrator.eval()

    cal_logits = calibrator(logits)
    torch.save(cal_logits, os.path.join(output_dir, "cal_logits.pt"))
    

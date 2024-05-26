
from typing import Literal
from affinecal.scripts.train import main as train
from affinecal.scripts.predict import main as predict


def main(
    task: Literal["train", "predict"],
    **kwargs,
):
    if task == "train":
        train(**kwargs)
    elif task == "predict":
        predict(**kwargs)
    else:
        raise ValueError(f"Invalid task: {task}")

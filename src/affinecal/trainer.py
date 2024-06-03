
import os
from typing import Literal, Union
import torch
from torch import nn, optim
from tqdm import tqdm

from .losses import init_loss


class Trainer:

    def __init__(
        self,
        accelerator: str = "cpu",
        learning_rate: float = 1.,
        max_ls: int = 40,
        epochs: int = 100,
        loss: Literal["cross_entropy", "brier"] = "cross_entropy",
        output_dir: Union[str, None] = None,
    ):
        self.device = torch.device(accelerator)
        self.learning_rate = learning_rate
        self.max_ls = max_ls
        self.epochs = epochs
        self.loss = init_loss(loss)
        self.output_dir = output_dir


    def init_state(self):
        self._state = {
            "global_step": 0,
            "current_epoch": 0,
            "state_dict": None,
            "kwargs": {
                "num_classes": None,
                "alpha": None,
                "beta": None,
            },
            "optimizer_state_dict": None,
            "train_loss_history": [],
            "val_loss_history": [],
            "best_val_loss": float("inf"),
        }

    def fit(
        self,
        model,
        train_logits: torch.Tensor,
        train_labels: torch.Tensor,
        val_logits: torch.Tensor = None,
        val_labels: torch.Tensor = None,
        ckpt_path: str = None,
    ):
        
        self.init_state()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.LBFGS(trainable_params, lr=self.learning_rate, max_iter=self.max_ls)
        model = model.to(self.device)
        self._state["kwargs"]["num_classes"] = model.num_classes
        self._state["kwargs"]["alpha"] = model.alpha
        self._state["kwargs"]["beta"] = model.beta

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        if ckpt_path is not None:
            self._state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(self._state["state_dict"])
            optimizer.load_state_dict(self._state["optimizer_state_dict"])

        def closure():
            cal_logits = model(train_logits)
            loss = self.loss(cal_logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            self._state["global_step"] += 1
            self._state["train_loss_history"].append(loss.item())
            return loss
        
        train_logits = train_logits.to(self.device, dtype=torch.float32)
        train_labels = train_labels.to(self.device, dtype=torch.long)
        if val_logits is not None:
            val_logits = val_logits.to(self.device, dtype=torch.float32)
            val_labels = val_labels.to(self.device, dtype=torch.long)

        model.train()
        try:
            for epoch in tqdm(range(self._state["current_epoch"], self.epochs), leave=False):
                loss = optimizer.step(closure)
                if val_logits is not None:
                    model.eval()
                    with torch.no_grad():
                        val_loss = self.loss(self(val_logits), val_labels).item()
                        self._state["val_loss_history"].append(val_loss)
                    model.train()
                    if val_loss < self._state["best_val_loss"]:
                        self._state["best_val_loss"] = val_loss
                        self.save_state(model, name="best")
                self._state["current_epoch"] += 1
                self._state["state_dict"] = model.state_dict()
                self._state["optimizer_state_dict"] = optimizer.state_dict()
                if self.output_dir is not None:
                    self.save_state(name="last")

        except KeyboardInterrupt:
            self._state["state_dict"] = model.state_dict()
            self._state["optimizer_state_dict"] = optimizer.state_dict()
            if self.output_dir is not None:
                self.save_state(name="last")

    def save_state(self, name = "best"):
        torch.save(self._state, os.path.join(self.output_dir, f"{name}.ckpt"))

        
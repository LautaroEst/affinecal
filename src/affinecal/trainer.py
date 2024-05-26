




import os
from typing import Literal
import torch
from torch import nn, optim

from .losses import init_loss


class Trainer:

    def __init__(
        self,
        accelerator: str = "cpu",
        learning_rate: float = 0.01,
        max_ls: int = 100,
        epochs: int = 100,
        loss: Literal["cross_entropy", "brier"] = "cross_entropy",
        output_dir: str = "./output",
    ):
        self.device = torch.device(accelerator)
        self.learning_rate = learning_rate
        self.max_ls = max_ls
        self.epochs = epochs
        self.loss = init_loss(loss)
        self.output_dir = output_dir

        self._state = {
            "global_step": 0,
            "current_epoch": 0,
            "state_dict": None,
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
        
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        optimizer = optim.LBFGS(trainable_params, lr=self.learning_rate, max_iter=self.max_ls)
        model = model.to(self.device)

        if ckpt_path is not None:
            self._state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(self._state["state_dict"])
            optimizer.load_state_dict(self._state["optimizer_state_dict"])

        def closure():
            cal_logits = self(train_logits)
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
            for epoch in range(self._state["current_epoch"], self.epochs):
                loss = optimizer.step(closure)
                if val_logits is not None:
                    model.eval()
                    with torch.no_grad():
                        val_loss = self.loss(self(val_logits), val_labels).item()
                        self._state["val_loss_history"].append(val_loss)
                    model.train()
                    if val_loss < self._state["best_val_loss"]:
                        self._state["best_val_loss"] = val_loss
                        self.save_model_state(model, name="best")
                self._state["current_epoch"] += 1
                self._state["state_dict"] = model.state_dict()
                self._state["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(self._state, os.path.join(self.output_dir, f"checkpoint-epoch{epoch}-step{self._state['global_step']}.ckpt"))
                self.save_model_state(model, name="last")
        except KeyboardInterrupt:
            self._state["state_dict"] = model.state_dict()
            self._state["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(self._state, os.path.join(self.output_dir, f"checkpoint-epoch{epoch}-step{self._state['global_step']}.ckpt"))
            self.save_model_state(model, name="last")

    def save_model_state(self, model, name = "best"):
        checkpoint = {
            "state_dict": model.state_dict(),
            "num_classes": model.num_classes,
            "alpha": model.alpha,
            "beta": model.beta,
        }
        torch.save(checkpoint, os.path.join(self.output_dir, f"{name}.pth"))

        
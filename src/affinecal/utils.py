
from .model import AffineCalibrator
from .trainer import Trainer
from .losses import init_loss
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
import numpy as np

def calibrate_xval(logits, targets, seed=0, condition_ids=None, stratified=True, nfolds=5, **kwargs):
    logitscal = torch.zeros(logits.size())
    
    if stratified:
        if condition_ids is not None:
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        else:
            # Use StratifiedKFold in this case for backward compatibility
            skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    else:
        if condition_ids is not None:
            skf = GroupKFold(n_splits=nfolds)
        else:
            skf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    alpha = kwargs.pop("alpha", "scalar")
    beta = kwargs.pop("beta", True)
    trainer = Trainer(**kwargs)
    for trni, tsti in skf.split(logits, targets, condition_ids):
        model = AffineCalibrator(num_classes=logits.shape[1], alpha=alpha, beta=beta)
        trainer.fit(model, logits[trni], targets[trni])
        with torch.no_grad():
            logitscal[tsti] = torch.log_softmax(model(logits[tsti]), dim=1)

    return logitscal

def train_on_test(logits, targets, **kwargs):
    alpha = kwargs.pop("alpha", "scalar")
    beta = kwargs.pop("beta", True)
    trainer = Trainer(**kwargs)
    model = AffineCalibrator(num_classes=logits.shape[1], alpha=alpha, beta=beta)
    trainer.fit(model, logits, targets)
    with torch.no_grad():
        logitscal = torch.log_softmax(model(logits), dim=1)
    if np.isnan(logitscal).any():
        print(logits, targets)
        print(model.alpha, model.beta)
    return logitscal


def cal_loss(logits, targets, relative=True, **kwargs):
    loss_name = kwargs.pop("loss", "cross_entropy")
    kwargs["loss"] = loss_name
    # logitscal = calibrate_xval(logits, targets, **kwargs)
    logitscal = train_on_test(logits, targets, **kwargs)
    loss_fn = init_loss(loss_name)
    start_loss = loss_fn(logits, targets)
    end_loss = loss_fn(logitscal, targets)
    if relative:
        return (start_loss - end_loss) / start_loss
    return start_loss - end_loss

def min_cal(logits, targets, **kwargs):
    loss_name = kwargs.pop("loss", "cross_entropy")
    kwargs["loss"] = loss_name
    logitscal = calibrate_xval(logits, targets, **kwargs)
    loss_fn = init_loss(loss_name)
    end_loss = loss_fn(logitscal, targets)
    return end_loss


    


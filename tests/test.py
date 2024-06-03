
from .expected_cost.calibration import train_calibrator, calibrate_scores, calibration_with_crossval

import torch

from affinecal import AffineCalibrator, Trainer, calibrate_xval, cal_loss

def main():
    torch.manual_seed(8349)

    N = 100
    c = 4
    logpost_trn = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_trn = torch.randint(0, c, (N,))
    
    logpost_tst = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_tst = torch.randint(0, c, (N,))

    model = train_calibrator(logpost_trn, targets_trn, calparams={"scale": False, "bias": True})
    callogpost = calibrate_scores(logpost_tst, model)
    print(model.temp, model.bias)


    model = AffineCalibrator(c, alpha="none", beta=True)
    trainer = Trainer(
        accelerator="cpu", 
        learning_rate=1.,
        max_ls=40,
        epochs=100,
        loss="cross_entropy",
        output_dir="./test_outputs"
    )
    trainer.fit(model, logpost_trn, targets_trn)
    callogpost = model(logpost_tst)
    print(model.alpha, model.beta)

def main():
    torch.manual_seed(8349)

    N = 100
    c = 4
    logpost_trn = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_trn = torch.randint(0, c, (N,))
    
    logpost_tst = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_tst = torch.randint(0, c, (N,))

    cal_logits = calibration_with_crossval(logpost_trn, targets_trn, calparams={"scale": True, "bias": True}, seed=0)
    print(cal_logits)
    cal_logits = calibrate_xval(logpost_trn, targets_trn, seed=0)
    print(cal_logits)

def main():
    torch.manual_seed(8349)

    N = 100
    c = 4
    logpost_trn = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_trn = torch.randint(0, c, (N,))
    
    logpost_tst = torch.log_softmax(torch.randn(N, c), dim=1)
    targets_tst = torch.randint(0, c, (N,))

    loss = cal_loss(logpost_tst, targets_tst, relative=True)
    print(loss)


    cal_logits = calibration_with_crossval(logpost_tst, targets_tst, calparams={"scale": True, "bias": True}, seed=0)
    start_loss = torch.nn.functional.cross_entropy(logpost_tst, targets_tst)
    end_loss = torch.nn.functional.cross_entropy(torch.from_numpy(cal_logits), targets_tst)
    print((end_loss - start_loss) / start_loss)






if __name__ == "__main__":
    main()
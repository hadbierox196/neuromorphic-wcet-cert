"""
Training loop for SeizureLIFNet on preprocessed spike tensors (§2.2.3).

- 30 epochs, Adam (lr=1e-3)
- cross-entropy on time-averaged output spike rates
- 70/10/20 stratified train/val/test split
- reports balanced accuracy, sensitivity/specificity, confusion matrix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import SeizureLIFNet, N_CHANNELS := 23  # noqa: F841 (kept for clarity)
from src.preprocessing import load_saved, WIN

EPOCHS = 30
LR = 1e-3
N_STEPS_TRAIN = 8  # reduced replay length for tractable training; see note below
SEED = 0

# NOTE ON N_STEPS: the paper's compiled inference function replays T_steps=512
# steps per §2.2.1/2.4 for the *compiled/WCET* analysis. For gradient-based
# training we replay a smaller number of steps (N_STEPS_TRAIN) and average
# output spike *rates* over that replay window, which is standard snnTorch
# practice and does not change the exported architecture (still exactly
# three LIFLayer nodes). Increase N_STEPS_TRAIN for closer fidelity to §2.2.3
# at the cost of training time.


def stratified_split(labels: np.ndarray, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split(idx):
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.10 * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    tr0, va0, te0 = split(idx0)
    tr1, va1, te1 = split(idx1)
    train = np.concatenate([tr0, tr1])
    val = np.concatenate([va0, va1])
    test = np.concatenate([te0, te1])
    rng.shuffle(train)
    return train, val, test


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return 0.5 * (sens + spec), sens, spec, dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


@torch.no_grad()
def evaluate(model: SeizureLIFNet, x: torch.Tensor, y: torch.Tensor, n_steps: int):
    model.eval()
    out = model.forward_sequence(x, n_steps=n_steps)  # (T, N, 2)
    rates = out.mean(dim=0)  # (N, 2)
    preds = rates.argmax(dim=1).cpu().numpy()
    ba, sens, spec, cm = balanced_accuracy(y.cpu().numpy(), preds)
    return ba, sens, spec, cm


def train(data_path: str, model_out: str, report_out: str,
          epochs: int = EPOCHS, n_steps: int = N_STEPS_TRAIN) -> dict:
    torch.manual_seed(SEED)
    ds = load_saved(data_path)
    x = ds.spikes.reshape(ds.spikes.shape[0], -1).astype(np.float32)  # (N, 23*512)
    y = ds.labels

    train_idx, val_idx, test_idx = stratified_split(y)
    x_t = torch.tensor(x)
    y_t = torch.tensor(y, dtype=torch.long)

    model = SeizureLIFNet()
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_ba": [], "val_ba": []}
    best_val_ba = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model.forward_sequence(x_t[train_idx], n_steps=n_steps)
        rates = out.mean(dim=0)
        loss = loss_fn(rates, y_t[train_idx])
        loss.backward()
        opt.step()

        train_ba, *_ = evaluate(model, x_t[train_idx], y_t[train_idx], n_steps)
        val_ba, *_ = evaluate(model, x_t[val_idx], y_t[val_idx], n_steps)
        history["train_ba"].append(train_ba)
        history["val_ba"].append(val_ba)
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            Path(model_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_out)
        print(f"epoch {epoch:02d}/{epochs}  loss={loss.item():.4f}  "
              f"train_ba={train_ba:.3f}  val_ba={val_ba:.3f}")

    model.load_state_dict(torch.load(model_out))
    test_ba, sens, spec, cm = evaluate(model, x_t[test_idx], y_t[test_idx], n_steps)

    report = {
        "test_balanced_accuracy": test_ba,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm,
        "best_val_balanced_accuracy": best_val_ba,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "epochs": epochs, "n_steps_train": n_steps,
    }
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="artifacts/build/spike_tensors.npz")
    ap.add_argument("--model-out", default="artifacts/build/best_model.pt")
    ap.add_argument("--report-out", default="artifacts/build/training_report.json")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--n-steps", type=int, default=N_STEPS_TRAIN)
    args = ap.parse_args()
    train(args.data, args.model_out, args.report_out, args.epochs, args.n_steps)


if __name__ == "__main__":
    main()

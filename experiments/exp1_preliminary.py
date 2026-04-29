"""
Experiment 1: Preliminary λ (pred_ann_weight) comparison.
Reproduces Table 1 (partial): λ=0 vs λ=0.05 with seed=42.

Expected result from paper:
  λ=0.00 → ~0.65 accuracy
  λ=0.05 → ~0.71 accuracy  (best)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import NMEDTDataset
from src.losses import CombinedLoss
from src.models import PredANNModel
from src.utils import evaluate_model, set_seed
from scripts.train import load_data, train_one_epoch

DATA_DIR = "data/processed"
OUTPUT_DIR = Path("results/exp1")
EPOCHS = 6000
BATCH_SIZE = 64
LR = 1e-3
SEED = 42
LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5]
LOG_EVERY = 500


def run_experiment(lambda_val: float, data_dir: str, seed: int, device: torch.device) -> float:
    set_seed(seed)
    train_eeg, train_audio, train_labels, val_eeg, val_audio, val_labels = load_data(data_dir)

    train_ds = NMEDTDataset(train_eeg, train_audio, train_labels)
    val_ds = NMEDTDataset(val_eeg, val_audio, val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PredANNModel(eeg_channels=train_eeg.shape[1]).to(device)
    criterion = CombinedLoss(pred_ann_weight=lambda_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % LOG_EVERY == 0:
            metrics = evaluate_model(model, val_loader, device)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
            print(f"  λ={lambda_val} epoch={epoch} val_acc={metrics['accuracy']:.4f}")

    return best_acc


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    results = []
    for lam in LAMBDAS:
        print(f"\n=== λ = {lam} ===")
        acc = run_experiment(lam, DATA_DIR, SEED, device)
        results.append({"lambda": lam, "accuracy": acc, "seed": SEED})
        print(f"  Best accuracy: {acc:.4f}")

    df = pd.DataFrame(results)
    print("\n=== Table 1 (partial) ===")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "table1_lambda_comparison.csv", index=False)

    with open(OUTPUT_DIR / "table1_lambda_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

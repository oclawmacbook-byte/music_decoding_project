"""
Training script for PredANN music identification from EEG.

Usage:
    python scripts/train.py \\
        --data_dir data/processed \\
        --model_type 2d \\
        --pred_ann_weight 0.05 \\
        --seed 42 \\
        --epochs 6000 \\
        --output_dir runs/exp1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import NMEDTDataset
from src.losses import CombinedLoss
from src.models import PredANNModel
from src.utils import evaluate_model, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PredANN model")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_type", choices=["1d", "2d"], default="2d")
    p.add_argument("--pred_ann_weight", type=float, default=0.05,
                   help="λ weight for PredANN loss (0 = classification only)")
    p.add_argument("--delay_ms", type=int, default=200,
                   help="EEG delay in ms (informational; data must be pre-processed with same value)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--output_dir", default="runs/default")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if omitted)")
    return p.parse_args()


def get_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(data_dir: str | Path) -> tuple:
    data_dir = Path(data_dir)
    train_eeg = np.load(data_dir / "train_eeg.npy")
    train_audio = np.load(data_dir / "train_audio.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    val_eeg = np.load(data_dir / "val_eeg.npy")
    val_audio = np.load(data_dir / "val_audio.npy")
    val_labels = np.load(data_dir / "val_labels.npy")
    return train_eeg, train_audio, train_labels, val_eeg, val_audio, val_labels


def train_one_epoch(
    model: PredANNModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"total": 0, "cls_eeg": 0, "cls_music": 0, "pred_ann": 0}
    n_batches = 0
    for eeg, audio, labels in loader:
        eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
        optimizer.zero_grad()
        z_eeg_I, z_music_I, z_eeg_II, z_music_II = model(eeg, audio)
        losses = criterion(z_eeg_I, z_music_I, z_eeg_II, z_music_II, labels)
        losses["total"].backward()
        optimizer.step()
        for k, v in losses.items():
            totals[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"λ (pred_ann_weight): {args.pred_ann_weight}")

    train_eeg, train_audio, train_labels, val_eeg, val_audio, val_labels = load_data(args.data_dir)
    eeg_channels = train_eeg.shape[1]

    train_ds = NMEDTDataset(train_eeg, train_audio, train_labels)
    val_ds = NMEDTDataset(val_eeg, val_audio, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PredANNModel(
        encoder_type=args.model_type,
        eeg_channels=eeg_channels,
    ).to(device)
    criterion = CombinedLoss(
        pred_ann_weight=args.pred_ann_weight,
        temperature=args.temperature,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % args.log_every == 0 or epoch == 1:
            val_metrics = evaluate_model(model, val_loader, device)
            elapsed = time.time() - t0
            row = {"epoch": epoch, "val_acc": val_metrics["accuracy"], **train_losses, "elapsed_s": elapsed}
            history.append(row)
            print(
                f"Epoch {epoch:5d} | val_acc={val_metrics['accuracy']:.4f} "
                f"| loss={train_losses['total']:.4f} "
                f"| cls_e={train_losses['cls_eeg']:.4f} "
                f"| pred={train_losses['pred_ann']:.4f} "
                f"| {elapsed:.0f}s"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "final_model.pt")

    results = {
        "best_val_acc": best_val_acc,
        "args": vars(args),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

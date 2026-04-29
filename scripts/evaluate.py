"""
Evaluate a trained PredANN model on the NMED-T test set.

Usage:
    python scripts/evaluate.py \\
        --model_path runs/exp1/best_model.pt \\
        --data_dir data/processed \\
        --model_type 2d \\
        --eval_length_s 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import NMEDTDataset
from src.models import PredANNModel
from src.utils import evaluate_model, per_class_accuracy, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_type", choices=["1d", "2d"], default="2d")
    p.add_argument("--eval_length_s", type=int, default=3,
                   help="Evaluation window length in seconds (3–7 for Table 5)")
    p.add_argument("--aggregation", choices=["mean", "max", "majority"], default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Optional JSON output path")
    return p.parse_args()


def get_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    data_dir = Path(args.data_dir)
    val_eeg = np.load(data_dir / "val_eeg.npy")
    val_audio = np.load(data_dir / "val_audio.npy")
    val_labels = np.load(data_dir / "val_labels.npy")
    subject_ids = np.load(data_dir / "val_subject_ids.npy") if (data_dir / "val_subject_ids.npy").exists() else None

    eeg_channels = val_eeg.shape[1]
    model = PredANNModel(encoder_type=args.model_type, eeg_channels=eeg_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    val_ds = NMEDTDataset(val_eeg, val_audio, val_labels)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_model(model, val_loader, device)
    print(f"Window-level accuracy: {metrics['accuracy']:.4f}")

    all_preds = []
    with torch.no_grad():
        for eeg, _, _ in val_loader:
            logits = model.predict_eeg(eeg.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
    all_preds = np.concatenate(all_preds)

    per_song = per_class_accuracy(all_preds, val_labels, n_classes=10)
    print("\nPer-song accuracy (Table 6):")
    for i, acc in enumerate(per_song):
        print(f"  Song {i+1}: {acc:.4f}")

    if subject_ids is not None:
        print("\nPer-subject accuracy (Table 7):")
        for subj in sorted(np.unique(subject_ids)):
            mask = subject_ids == subj
            subj_acc = (all_preds[mask] == val_labels[mask]).mean()
            print(f"  Subject {subj:02d}: {subj_acc:.4f}")

    results = {
        "window_accuracy": metrics["accuracy"],
        "per_song_accuracy": per_song.tolist(),
        "args": vars(args),
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

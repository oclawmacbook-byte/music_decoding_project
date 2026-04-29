from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Window-level classification accuracy using EEG encoder only."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg, audio, labels in dataloader:
            eeg, labels = eeg.to(device), labels.to(device)
            logits = model.predict_eeg(eeg)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


def evaluate_sliding_window(
    model: torch.nn.Module,
    eeg_seq: np.ndarray,
    window_size: int,
    stride: int,
    device: torch.device,
    n_classes: int = 10,
    method: Literal["mean", "max", "majority"] = "mean",
) -> int:
    """
    Classify a full EEG sequence by aggregating window predictions.
    Returns predicted song label (0-9).
    """
    model.eval()
    _, n_samples = eeg_seq.shape
    window_logits = []

    with torch.no_grad():
        for start in range(0, n_samples - window_size + 1, stride):
            window = torch.from_numpy(eeg_seq[:, start: start + window_size]).float()
            window = window.unsqueeze(0).to(device)   # (1, channels, time)
            logits = model.predict_eeg(window)         # (1, n_classes)
            window_logits.append(logits.squeeze(0).cpu().numpy())

    if not window_logits:
        return 0

    logits_arr = np.stack(window_logits)  # (n_windows, n_classes)

    if method == "mean":
        return int(logits_arr.mean(axis=0).argmax())
    elif method == "max":
        return int(logits_arr.max(axis=0).argmax())
    elif method == "majority":
        preds = logits_arr.argmax(axis=1)
        counts = np.bincount(preds, minlength=n_classes)
        return int(counts.argmax())
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def evaluate_sequences(
    model: torch.nn.Module,
    eeg_sequences: list[np.ndarray],
    labels: list[int],
    window_size: int,
    stride: int,
    device: torch.device,
    method: Literal["mean", "max", "majority"] = "mean",
) -> dict[str, float]:
    """Evaluate over full-length sequences and return accuracy."""
    correct = 0
    for eeg_seq, true_label in zip(eeg_sequences, labels):
        pred = evaluate_sliding_window(
            model, eeg_seq, window_size, stride, device, method=method
        )
        if pred == true_label:
            correct += 1
    return {"accuracy": correct / len(labels) if labels else 0.0}


def mcnemar_test(predictions1: np.ndarray, predictions2: np.ndarray, labels: np.ndarray) -> dict:
    """
    McNemar's test for paired nominal data — used to compare model variants.
    Returns chi2 statistic and p-value.
    """
    from scipy.stats import chi2

    correct1 = predictions1 == labels
    correct2 = predictions2 == labels

    # b: model1 correct, model2 wrong; c: model1 wrong, model2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": 0, "c": 0}

    # With continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return {"chi2": chi2_stat, "p_value": p_value, "b": int(b), "c": int(c)}


def per_class_accuracy(
    predictions: np.ndarray, labels: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    accs = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            accs[c] = (predictions[mask] == labels[mask]).mean()
    return accs


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

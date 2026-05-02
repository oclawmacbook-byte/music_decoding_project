"""
Generate a synthetic dummy dataset for testing the PredANN pipeline
without requiring real NMED-T data.

Output files match exactly what preprocess_data.py produces:
  data/dummy/
    train_eeg.npy          (N_train, 128, 375)  float32
    train_audio.npy        (N_train, 66150)      float32
    train_labels.npy       (N_train,)            int64
    train_subject_ids.npy  (N_train,)            int32
    val_eeg.npy            (N_val,   128, 375)   float32
    val_audio.npy          (N_val,   66150)       float32
    val_labels.npy         (N_val,)              int64
    val_subject_ids.npy    (N_val,)              int32
    meta.json

Usage:
    python scripts/create_dummy_dataset.py
    python scripts/create_dummy_dataset.py --out_dir data/dummy --n_subjects 3 --windows_per_song 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Must match preprocessing.py constants
N_CHANNELS = 128
WINDOW_SIZE = 375       # 3 s at 125 Hz
AUDIO_SR = 22050
EEG_SR = 125
AUDIO_WINDOW_SIZE = int(WINDOW_SIZE * AUDIO_SR / EEG_SR)   # 66150
N_SONGS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create dummy dataset for testing")
    p.add_argument("--out_dir", default="data/dummy")
    p.add_argument("--n_subjects", type=int, default=5,
                   help="Number of synthetic subjects (default: 5)")
    p.add_argument("--windows_per_song", type=int, default=8,
                   help="Training windows per (subject, song) pair (default: 8)")
    p.add_argument("--val_ratio", type=float, default=0.25,
                   help="Fraction of windows used for validation (default: 0.25)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_eeg_window(rng: np.random.Generator, song_id: int) -> np.ndarray:
    """Synthetic EEG: Gaussian noise with a per-song mean shift for separability."""
    signal = rng.standard_normal((N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
    signal += (song_id - N_SONGS / 2) * 0.1   # small class-specific bias
    return signal


def make_audio_window(rng: np.random.Generator, song_id: int) -> np.ndarray:
    """Synthetic audio: sum of sinusoids with song-specific fundamental frequency."""
    t = np.linspace(0, WINDOW_SIZE / EEG_SR, AUDIO_WINDOW_SIZE, dtype=np.float32)
    freq = 220.0 * (2 ** (song_id / 12))       # chromatic scale starting at A3
    audio = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.1 * rng.standard_normal(AUDIO_WINDOW_SIZE).astype(np.float32)
    )
    return audio.astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(args.windows_per_song * args.val_ratio))
    n_train = args.windows_per_song - n_val

    train_eeg, train_audio, train_labels, train_sids = [], [], [], []
    val_eeg,   val_audio,   val_labels,   val_sids   = [], [], [], []

    for subj in range(1, args.n_subjects + 1):
        for song_id in range(N_SONGS):
            for _ in range(n_train):
                train_eeg.append(make_eeg_window(rng, song_id))
                train_audio.append(make_audio_window(rng, song_id))
                train_labels.append(song_id)
                train_sids.append(subj)

            for _ in range(n_val):
                val_eeg.append(make_eeg_window(rng, song_id))
                val_audio.append(make_audio_window(rng, song_id))
                val_labels.append(song_id)
                val_sids.append(subj)

    np.save(out_dir / "train_eeg.npy",         np.stack(train_eeg))
    np.save(out_dir / "train_audio.npy",        np.stack(train_audio))
    np.save(out_dir / "train_labels.npy",       np.array(train_labels, dtype=np.int64))
    np.save(out_dir / "train_subject_ids.npy",  np.array(train_sids,   dtype=np.int32))

    np.save(out_dir / "val_eeg.npy",            np.stack(val_eeg))
    np.save(out_dir / "val_audio.npy",          np.stack(val_audio))
    np.save(out_dir / "val_labels.npy",         np.array(val_labels,   dtype=np.int64))
    np.save(out_dir / "val_subject_ids.npy",    np.array(val_sids,     dtype=np.int32))

    meta = {
        "dummy": True,
        "delay_ms": 200,
        "delay_samples": 25,
        "seed": args.seed,
        "n_subjects": args.n_subjects,
        "n_songs": N_SONGS,
        "windows_per_song": args.windows_per_song,
        "n_train_per_song": n_train,
        "n_val_per_song": n_val,
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "eeg_shape": [N_CHANNELS, WINDOW_SIZE],
        "audio_window_size": AUDIO_WINDOW_SIZE,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dummy dataset saved to: {out_dir}")
    print(f"  Train: {meta['train_size']} windows  "
          f"(EEG: {N_CHANNELS}ch × {WINDOW_SIZE} samples, "
          f"audio: {AUDIO_WINDOW_SIZE} samples)")
    print(f"  Val:   {meta['val_size']} windows")
    print()
    print("Quick-start training command:")
    print(f"  python scripts/train.py --data_dir {out_dir} "
          f"--epochs 50 --batch_size 32 --output_dir runs/dummy_test")


if __name__ == "__main__":
    main()

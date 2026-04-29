"""
Preprocess raw NMED-T data into windowed numpy arrays ready for training.

Usage:
    python scripts/preprocess_data.py --raw_dir /path/to/nmedt --out_dir data/processed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    DELAY_SAMPLES,
    TARGET_SR,
    TRAIN_STRIDE,
    VAL_STRIDE,
    downsample_eeg,
    extract_windows,
    load_nmedt_audio,
    load_nmedt_eeg,
    normalize_eeg,
    split_dataset,
    truncate_to_max_duration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="Path to raw NMED-T data directory")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 21)),
                   help="Subject IDs to process (default: all 20)")
    p.add_argument("--delay_ms", type=int, default=200,
                   help="EEG-to-audio delay in ms (default: 200)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delay_samples = int(args.delay_ms * TARGET_SR / 1000)
    audio_dict = load_nmedt_audio(args.raw_dir)

    all_train_eeg, all_train_audio, all_train_labels = [], [], []
    all_val_eeg, all_val_audio, all_val_labels = [], [], []
    subject_ids_train, subject_ids_val = [], []

    for subj in args.subjects:
        print(f"Processing subject {subj:02d}...")
        eeg_raw, song_sequence = load_nmedt_eeg(args.raw_dir, subj)
        eeg_ds = downsample_eeg(eeg_raw)
        eeg_norm = normalize_eeg(eeg_ds)
        eeg_trunc = truncate_to_max_duration(eeg_norm)

        all_eeg, all_audio, all_labels = [], [], []
        for song_id in np.unique(song_sequence):
            song_audio = audio_dict[int(song_id)]
            eeg_w, audio_w, lbl = extract_windows(
                eeg_trunc, song_audio, int(song_id),
                stride=TRAIN_STRIDE, delay_samples=delay_samples,
            )
            all_eeg.append(eeg_w)
            all_audio.append(audio_w)
            all_labels.append(lbl)

        eeg_all = np.concatenate(all_eeg)
        audio_all = np.concatenate(all_audio)
        labels_all = np.concatenate(all_labels)

        eeg_tr, eeg_v, aud_tr, aud_v, lbl_tr, lbl_v = split_dataset(
            eeg_all, audio_all, labels_all, seed=args.seed
        )

        all_train_eeg.append(eeg_tr)
        all_train_audio.append(aud_tr)
        all_train_labels.append(lbl_tr)
        subject_ids_train.append(np.full(len(lbl_tr), subj, dtype=np.int32))

        all_val_eeg.append(eeg_v)
        all_val_audio.append(aud_v)
        all_val_labels.append(lbl_v)
        subject_ids_val.append(np.full(len(lbl_v), subj, dtype=np.int32))

    np.save(out_dir / "train_eeg.npy", np.concatenate(all_train_eeg))
    np.save(out_dir / "train_audio.npy", np.concatenate(all_train_audio))
    np.save(out_dir / "train_labels.npy", np.concatenate(all_train_labels))
    np.save(out_dir / "train_subject_ids.npy", np.concatenate(subject_ids_train))

    np.save(out_dir / "val_eeg.npy", np.concatenate(all_val_eeg))
    np.save(out_dir / "val_audio.npy", np.concatenate(all_val_audio))
    np.save(out_dir / "val_labels.npy", np.concatenate(all_val_labels))
    np.save(out_dir / "val_subject_ids.npy", np.concatenate(subject_ids_val))

    meta = {
        "delay_ms": args.delay_ms,
        "delay_samples": delay_samples,
        "seed": args.seed,
        "subjects": args.subjects,
        "train_size": int(np.concatenate(all_train_labels).shape[0]),
        "val_size": int(np.concatenate(all_val_labels).shape[0]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved preprocessed data to {out_dir}")
    print(f"  Train: {meta['train_size']} windows")
    print(f"  Val:   {meta['val_size']} windows")


if __name__ == "__main__":
    main()

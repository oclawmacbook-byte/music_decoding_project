"""
Download NMED-T cleaned EEG data for one subject and prepare for training.

The NMED-T cleaned files are organized by song (songSS_Imputed.mat), each
containing EEG from all 20 participants with shape (128, T, 20) at 125Hz.

Audio stimuli are not published due to copyright, so this script generates
synthetic sinusoidal audio (per-song unique tones) as a stand-in. This allows
the full pipeline to run; the contrastive loss won't reflect real EEG-audio
alignment, but classification loss and pipeline correctness can be verified.

Usage:
    uv run scripts/download_nmedt_subject.py --subject 1 --out_dir data/nmedt_s01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

# NMED-T constants
TARGET_SR = 125          # Hz — cleaned files are already at 125Hz
AUDIO_SR = 22050         # Hz — audio sample rate expected by model
WINDOW_SIZE = 375        # 3s × 125Hz
AUDIO_WINDOW_SIZE = 66150  # 3s × 22050Hz
TRAIN_STRIDE = 200
DELAY_SAMPLES = 25       # 200ms × 125Hz
MAX_DURATION_S = 240     # 4 minutes
SKIP_SECONDS = 15        # offset used in original NMED-T analysis

# Stanford Digital Repository base URL
SDR_BASE = "https://stacks.stanford.edu/file/druid:jn859kj8079"

# Song triggers 21-30 → song labels 0-9
SONG_TRIGGERS = list(range(21, 31))

# Participant mapping: clean index (1-based) → raw file number
CLEAN_TO_RAW = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
    11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 19, 18: 20,
    19: 21, 20: 23,
}


def download_file(url: str, dest: Path, desc: str = "") -> None:
    desc = desc or dest.name
    print(f"  Downloading {desc} ...", flush=True)

    def reporthook(count, block_size, total_size):
        pct = count * block_size / total_size * 100 if total_size > 0 else 0
        mb = count * block_size / 1e6
        print(f"\r  {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print(f"\r  Done  {dest.stat().st_size / 1e6:.0f} MB", flush=True)


def extract_subject_eeg(mat_path: Path, trigger: int, subject_idx: int) -> np.ndarray:
    """
    Load songSS_Imputed.mat and return EEG for one participant.
    Variable name inside: data{trigger}, shape (128, T, 20).
    Returns (128, T_truncated) float32 array.
    """
    print(f"  Loading {mat_path.name} ...", flush=True)
    data = scipy.io.loadmat(str(mat_path))
    key = f"data{trigger}"
    eeg = np.array(data[key], dtype=np.float32)  # (128, T, 20)

    # Select participant (0-indexed)
    eeg_subj = eeg[:, :, subject_idx]  # (128, T)

    # Skip first SKIP_SECONDS (15s) and take up to MAX_DURATION_S (240s)
    skip = SKIP_SECONDS * TARGET_SR
    end = skip + MAX_DURATION_S * TARGET_SR
    eeg_subj = eeg_subj[:, skip:end]

    return eeg_subj


def normalize_eeg(eeg: np.ndarray, clamp: float = 20.0) -> np.ndarray:
    scaler = RobustScaler()
    eeg_norm = scaler.fit_transform(eeg.T).T
    return np.clip(eeg_norm, -clamp, clamp).astype(np.float32)


def make_synthetic_audio(song_label: int, n_samples: int, sr: int = AUDIO_SR) -> np.ndarray:
    """
    Generate unique sinusoidal audio for each song label (placeholder for real audio).
    Uses a chord of frequencies derived from the song index.
    """
    t = np.arange(n_samples) / sr
    base_freq = 110 * (2 ** (song_label / 12))  # A2 shifted by song semitones
    audio = (
        np.sin(2 * np.pi * base_freq * t) +
        0.5 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.25 * np.sin(2 * np.pi * base_freq * 3 * t)
    ).astype(np.float32)
    audio /= np.abs(audio).max() + 1e-8
    return audio


def extract_windows(
    eeg: np.ndarray, song_audio: np.ndarray, song_label: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_channels, n_samples = eeg.shape
    eeg_windows, audio_windows, labels = [], [], []

    for start in range(0, n_samples - WINDOW_SIZE - DELAY_SAMPLES + 1, TRAIN_STRIDE):
        eeg_win = eeg[:, start + DELAY_SAMPLES: start + DELAY_SAMPLES + WINDOW_SIZE]
        audio_start = int(start * AUDIO_SR / TARGET_SR)
        audio_end = audio_start + AUDIO_WINDOW_SIZE
        if audio_end > len(song_audio):
            break
        audio_win = song_audio[audio_start:audio_end]
        eeg_windows.append(eeg_win)
        audio_windows.append(audio_win)
        labels.append(song_label)

    if not eeg_windows:
        return (
            np.empty((0, n_channels, WINDOW_SIZE), dtype=np.float32),
            np.empty((0, AUDIO_WINDOW_SIZE), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.stack(eeg_windows).astype(np.float32),
        np.stack(audio_windows).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1,
                   help="Clean subject index (1-20, default: 1)")
    p.add_argument("--out_dir", default="data/nmedt_s01")
    p.add_argument("--cache_dir", default="data/nmedt_cache",
                   help="Directory to cache/reuse downloaded .mat files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_cache", action="store_true",
                   help="Keep downloaded .mat files after extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    subject_idx = args.subject - 1  # 0-indexed for numpy slicing
    raw_id = CLEAN_TO_RAW[args.subject]
    print(f"Subject: clean={args.subject}, raw participant={raw_id:02d}")

    all_eeg, all_audio, all_labels = [], [], []

    for song_label, trigger in enumerate(SONG_TRIGGERS):
        fname = f"song{trigger}_Imputed.mat"
        url = f"{SDR_BASE}/{fname}"
        mat_path = cache_dir / fname

        if not mat_path.exists():
            download_file(url, mat_path, desc=f"{fname} (song {song_label+1}/10)")
        else:
            print(f"  Using cached {fname}")

        eeg = extract_subject_eeg(mat_path, trigger, subject_idx)
        eeg = normalize_eeg(eeg)

        n_audio_samples = int(eeg.shape[1] * AUDIO_SR / TARGET_SR) + AUDIO_WINDOW_SIZE
        song_audio = make_synthetic_audio(song_label, n_audio_samples)

        eeg_w, audio_w, lbls = extract_windows(eeg, song_audio, song_label)
        print(f"    Song {song_label+1}: {len(lbls)} windows extracted")

        all_eeg.append(eeg_w)
        all_audio.append(audio_w)
        all_labels.append(lbls)

        if not args.keep_cache:
            mat_path.unlink()
            print(f"    Deleted {fname} to free disk space")

    eeg_all = np.concatenate(all_eeg)
    audio_all = np.concatenate(all_audio)
    labels_all = np.concatenate(all_labels)

    print(f"\nTotal windows: {len(labels_all)}")

    idx = np.arange(len(labels_all))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.25, random_state=args.seed, stratify=labels_all
    )

    np.save(out_dir / "train_eeg.npy", eeg_all[train_idx])
    np.save(out_dir / "train_audio.npy", audio_all[train_idx])
    np.save(out_dir / "train_labels.npy", labels_all[train_idx])
    np.save(out_dir / "train_subject_ids.npy", np.full(len(train_idx), args.subject, dtype=np.int32))

    np.save(out_dir / "val_eeg.npy", eeg_all[val_idx])
    np.save(out_dir / "val_audio.npy", audio_all[val_idx])
    np.save(out_dir / "val_labels.npy", labels_all[val_idx])
    np.save(out_dir / "val_subject_ids.npy", np.full(len(val_idx), args.subject, dtype=np.int32))

    meta = {
        "subject_clean": args.subject,
        "subject_raw": raw_id,
        "seed": args.seed,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "note": "Audio is synthetic (sinusoidal placeholder); real NMED-T audio is not published.",
        "window_size": WINDOW_SIZE,
        "delay_samples": DELAY_SAMPLES,
        "target_sr": TARGET_SR,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  Train: {meta['train_size']} windows")
    print(f"  Val:   {meta['val_size']} windows")
    print(f"\nNote: Audio is synthetic. Real NMED-T audio is copyright-protected and unpublished.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# NMED-T dataset constants
ORIG_SR = 1000       # Hz
TARGET_SR = 125      # Hz
MAX_DURATION_S = 240  # 4 minutes shared across all recordings
WINDOW_SIZE = 375    # samples at 125Hz = 3 seconds
TRAIN_STRIDE = 200
VAL_STRIDE = 1
DELAY_SAMPLES = 25   # 200ms at 125Hz — empirically optimal (Table 2)


def load_nmedt_eeg(data_dir: str | Path, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EEG and stimulus timing from a single NMED-T subject .mat file.
    Returns (eeg, song_labels) where eeg shape is (n_channels, n_samples).
    """
    data_dir = Path(data_dir)
    mat_path = data_dir / f"s{subject_id:02d}.mat"
    data = scipy.io.loadmat(str(mat_path))

    # NMED-T stores EEG under key 'EEG' or 'eeg'; adapt as needed
    eeg_key = next(k for k in data if not k.startswith("_") and "eeg" in k.lower())
    eeg = np.array(data[eeg_key], dtype=np.float32)  # (channels, samples)

    label_key = next(k for k in data if not k.startswith("_") and "label" in k.lower())
    labels = np.array(data[label_key]).flatten().astype(int)

    return eeg, labels


def _load_audio_from_mat(path: Path, target_sr: int = 22050) -> np.ndarray:
    """Extract audio waveform from a .mat file and resample to target_sr."""
    import librosa

    data = scipy.io.loadmat(str(path))
    # NED-T stores audio under keys like 'audio', 'waveform', 'data', 'stim'
    audio_keys = [k for k in data if not k.startswith("_")]
    # Prefer keys containing recognisable audio-related names
    preferred = [k for k in audio_keys if any(kw in k.lower() for kw in ("audio", "wav", "stim", "data", "wave"))]
    key = preferred[0] if preferred else audio_keys[0]
    audio = np.array(data[key], dtype=np.float64).squeeze()

    # If a sampling-rate field exists, use it; otherwise assume 44100 Hz
    sr_keys = [k for k in data if not k.startswith("_") and "sr" in k.lower()]
    src_sr = int(np.array(data[sr_keys[0]]).flat[0]) if sr_keys else 44100

    if src_sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=target_sr)
    return audio.astype(np.float32)


def load_nmedt_audio(data_dir: str | Path) -> dict[int, np.ndarray]:
    """Load the 10 song audio clips from NMED-T (.mat, WAV, or NPY)."""
    import librosa

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    songs: dict[int, np.ndarray] = {}
    for song_idx in range(10):
        # Search order: .mat first (native NED-T format), then .npy/.wav/etc.
        candidates = (
            list(audio_dir.glob(f"song_{song_idx:02d}.mat")) +
            list(audio_dir.glob(f"{song_idx:02d}.mat")) +
            list(audio_dir.glob(f"song_{song_idx:02d}.*")) +
            list(audio_dir.glob(f"{song_idx:02d}.*"))
        )
        if not candidates:
            raise FileNotFoundError(f"No audio file found for song {song_idx} in {audio_dir}")
        path = candidates[0]
        if path.suffix == ".mat":
            audio = _load_audio_from_mat(path)
        elif path.suffix == ".npy":
            audio = np.load(str(path)).astype(np.float32)
        else:
            audio, _ = librosa.load(str(path), sr=22050, mono=True)
        songs[song_idx] = audio
    return songs


def downsample_eeg(eeg: np.ndarray, orig_sr: int = ORIG_SR, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample EEG from orig_sr to target_sr along the time axis (axis=1 for 2D input)."""
    n_samples_out = int(eeg.shape[1] * target_sr / orig_sr)
    return scipy.signal.resample(eeg, n_samples_out, axis=1).astype(np.float32)


def normalize_eeg(eeg: np.ndarray, clamp: float = 20.0) -> np.ndarray:
    """
    RobustScaler per channel (subtract median, divide by IQR) then clamp to ±clamp.
    Input shape: (n_channels, n_samples).
    """
    scaler = RobustScaler()
    # RobustScaler expects (samples, features); transpose, fit, transpose back
    eeg_norm = scaler.fit_transform(eeg.T).T
    return np.clip(eeg_norm, -clamp, clamp).astype(np.float32)


def truncate_to_max_duration(
    eeg: np.ndarray,
    sr: int = TARGET_SR,
    max_duration_s: int = MAX_DURATION_S,
) -> np.ndarray:
    max_samples = max_duration_s * sr
    return eeg[:, :max_samples]


def extract_windows(
    eeg: np.ndarray,
    audio_segment: np.ndarray,
    song_label: int,
    window_size: int = WINDOW_SIZE,
    stride: int = TRAIN_STRIDE,
    delay_samples: int = DELAY_SAMPLES,
    audio_window_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over the EEG and corresponding audio segment.
    The 200ms delay shifts EEG onset relative to audio: EEG[t+delay] aligns with audio[t].
    Returns (eeg_windows, audio_windows, labels).
    """
    n_channels, n_samples = eeg.shape
    if audio_window_size is None:
        # Map EEG window size to audio samples at 22050Hz
        audio_window_size = int(window_size * 22050 / TARGET_SR)

    eeg_windows, audio_windows, labels = [], [], []
    audio_sr = 22050
    n_audio_samples = len(audio_segment)

    for start in range(0, n_samples - window_size - delay_samples + 1, stride):
        eeg_win = eeg[:, start + delay_samples: start + delay_samples + window_size]

        # Corresponding audio segment (before delay onset)
        audio_start = int(start * audio_sr / TARGET_SR)
        audio_end = audio_start + audio_window_size
        if audio_end > n_audio_samples:
            break
        audio_win = audio_segment[audio_start:audio_end]

        eeg_windows.append(eeg_win)
        audio_windows.append(audio_win)
        labels.append(song_label)

    if not eeg_windows:
        return np.empty((0, n_channels, window_size), dtype=np.float32), \
               np.empty((0, audio_window_size), dtype=np.float32), \
               np.empty(0, dtype=np.int64)

    return (
        np.stack(eeg_windows).astype(np.float32),
        np.stack(audio_windows).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def split_dataset(
    eeg_windows: np.ndarray,
    audio_windows: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split to maintain class balance."""
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        eeg_windows[train_idx], eeg_windows[val_idx],
        audio_windows[train_idx], audio_windows[val_idx],
        labels[train_idx], labels[val_idx],
    )


def preprocess_subject(
    data_dir: str | Path,
    subject_id: int,
    audio_dict: dict[int, np.ndarray],
    stride: int = TRAIN_STRIDE,
    delay_samples: int = DELAY_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline for a single subject."""
    eeg_raw, song_sequence = load_nmedt_eeg(data_dir, subject_id)
    eeg_ds = downsample_eeg(eeg_raw, ORIG_SR, TARGET_SR)
    eeg_norm = normalize_eeg(eeg_ds)
    eeg_trunc = truncate_to_max_duration(eeg_norm)

    all_eeg, all_audio, all_labels = [], [], []
    unique_songs = np.unique(song_sequence)
    for song_id in unique_songs:
        song_mask = song_sequence == song_id
        # Assume EEG columns correspond to song segments sequentially
        # Adapt indexing based on actual NMED-T format
        song_eeg = eeg_trunc[:, :MAX_DURATION_S * TARGET_SR]
        song_audio = audio_dict[int(song_id)]

        eeg_w, audio_w, labels = extract_windows(
            song_eeg, song_audio, int(song_id),
            stride=stride, delay_samples=delay_samples,
        )
        all_eeg.append(eeg_w)
        all_audio.append(audio_w)
        all_labels.append(labels)

    return (
        np.concatenate(all_eeg),
        np.concatenate(all_audio),
        np.concatenate(all_labels),
    )

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NMEDTDataset(Dataset):
    """
    PyTorch Dataset for NMED-T windowed EEG/audio pairs.

    eeg_data:   (N, n_channels, window_size) float32
    audio_data: (N, audio_window_size) float32  — raw waveform
    labels:     (N,) int64
    """

    def __init__(
        self,
        eeg_data: np.ndarray,
        audio_data: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        assert len(eeg_data) == len(audio_data) == len(labels)
        self.eeg = torch.from_numpy(eeg_data)
        self.audio = torch.from_numpy(audio_data)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg = self.eeg[idx]
        audio = self.audio[idx]
        label = self.labels[idx]
        if self.transform is not None:
            eeg, audio = self.transform(eeg, audio)
        return eeg, audio, label


class SubjectDataset(NMEDTDataset):
    """Dataset filtered to a single subject for per-subject evaluation."""

    @classmethod
    def from_full(
        cls,
        eeg_data: np.ndarray,
        audio_data: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        subject_id: int,
        **kwargs,
    ) -> "SubjectDataset":
        mask = subject_ids == subject_id
        return cls(eeg_data[mask], audio_data[mask], labels[mask], **kwargs)

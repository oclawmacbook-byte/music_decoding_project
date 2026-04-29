from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class Conv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN2DEncoder(nn.Module):
    """
    2D CNN encoder adapted from SampleCNN (Lee et al. 2018).
    Input shape: (batch, channels, time) → reshaped to (batch, 1, channels, time).
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            Conv2DBlock(in_channels, 32),
            Conv2DBlock(32, 64),
            Conv2DBlock(64, 128),
        )
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, height, width)
        x = self.convs(x)
        x = self.dropout(x)
        x = self.pool(x)          # (batch, 128, 1, 1)
        x = x.flatten(1)          # (batch, 128)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DEncoder(nn.Module):
    """1D CNN encoder for ablation comparison (Table 3)."""
    def __init__(self, in_channels: int = 128, embed_dim: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            Conv1DBlock(in_channels, 32),
            Conv1DBlock(32, 64),
            Conv1DBlock(64, 128),
        )
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.convs(x)
        x = self.dropout(x)
        x = self.pool(x)   # (batch, 128, 1)
        x = x.flatten(1)   # (batch, 128)
        return x


class Projector(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 100, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelSpecEncoder(nn.Module):
    """Wrapper that computes mel spectrogram then passes through 2D CNN."""
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db = T.AmplitudeToDB()
        self.encoder = CNN2DEncoder(in_channels=1, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_samples)
        spec = self.mel(x)           # (batch, n_mels, frames)
        spec = self.db(spec)
        spec = spec.unsqueeze(1)     # (batch, 1, n_mels, frames)
        return self.encoder(spec)


class PredANNModel(nn.Module):
    """
    Full PredANN model with dual encoders (EEG + music) and dual projectors each.
    Projector I → classification (10-class)
    Projector II → contrastive alignment with PredANN loss
    """
    def __init__(
        self,
        encoder_type: str = "2d",
        embed_dim: int = 128,
        n_classes: int = 10,
        eeg_channels: int = 128,
        audio_sample_rate: int = 22050,
        n_mels: int = 64,
    ):
        super().__init__()
        assert encoder_type in ("1d", "2d"), "encoder_type must be '1d' or '2d'"
        self.encoder_type = encoder_type

        if encoder_type == "2d":
            self.eeg_encoder = CNN2DEncoder(in_channels=1, embed_dim=embed_dim)
            self.music_encoder = MelSpecEncoder(
                sample_rate=audio_sample_rate,
                n_mels=n_mels,
                embed_dim=embed_dim,
            )
        else:
            self.eeg_encoder = CNN1DEncoder(in_channels=eeg_channels, embed_dim=embed_dim)
            self.music_encoder = CNN1DEncoder(in_channels=1, embed_dim=embed_dim)

        self.eeg_proj_I = Projector(embed_dim, 100, n_classes)
        self.eeg_proj_II = Projector(embed_dim, 100, 100)
        self.music_proj_I = Projector(embed_dim, 100, n_classes)
        self.music_proj_II = Projector(embed_dim, 100, 100)

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "2d":
            # (batch, channels, time) → (batch, 1, channels, time)
            eeg = eeg.unsqueeze(1)
        return self.eeg_encoder(eeg)

    def encode_music(self, audio: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "2d":
            return self.music_encoder(audio)   # MelSpecEncoder handles unsqueeze
        else:
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)     # (batch, 1, time)
            return self.music_encoder(audio)

    def forward(
        self, eeg: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_eeg = self.encode_eeg(eeg)
        h_music = self.encode_music(audio)

        z_eeg_I = self.eeg_proj_I(h_eeg)
        z_eeg_II = self.eeg_proj_II(h_eeg)
        z_music_I = self.music_proj_I(h_music)
        z_music_II = self.music_proj_II(h_music)

        return z_eeg_I, z_music_I, z_eeg_II, z_music_II

    def predict_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference: classify from EEG only."""
        h = self.encode_eeg(eeg)
        return self.eeg_proj_I(h)

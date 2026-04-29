import torch
import torch.nn as nn
import torch.nn.functional as F


class PredANNLoss(nn.Module):
    """
    InfoNCE-style contrastive loss between EEG and music embeddings.
    Stop-gradient on music branch means only the EEG encoder learns from this loss,
    preventing the music encoder from collapsing to match EEG representations.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_eeg_II: torch.Tensor, z_music_II: torch.Tensor) -> torch.Tensor:
        z_music_II = z_music_II.detach()  # stop-gradient on music branch

        z_eeg = F.normalize(z_eeg_II, dim=-1)
        z_music = F.normalize(z_music_II, dim=-1)

        sim = torch.matmul(z_eeg, z_music.T) / self.temperature

        batch_size = z_eeg.size(0)
        labels = torch.arange(batch_size, device=z_eeg.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, pred_ann_weight: float = 0.05, temperature: float = 0.07):
        super().__init__()
        self.pred_ann_weight = pred_ann_weight
        self.ce = nn.CrossEntropyLoss()
        self.pred_ann = PredANNLoss(temperature=temperature)

    def forward(
        self,
        logits_eeg: torch.Tensor,
        logits_music: torch.Tensor,
        z_eeg_II: torch.Tensor,
        z_music_II: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_cls_e = self.ce(logits_eeg, labels)
        l_cls_m = self.ce(logits_music, labels)
        l_pred = self.pred_ann(z_eeg_II, z_music_II)
        total = l_cls_e + l_cls_m + self.pred_ann_weight * l_pred
        return {
            "total": total,
            "cls_eeg": l_cls_e,
            "cls_music": l_cls_m,
            "pred_ann": l_pred,
        }

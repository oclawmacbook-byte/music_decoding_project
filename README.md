# Music Decoding Project

Reproduction of **"Predicting artificial neural network representations to learn recognition model for music identification from brain recordings"** (Akama et al., *Scientific Reports* 2025, DOI: [10.1038/s41598-025-02790-6](https://doi.org/10.1038/s41598-025-02790-6)).

## Overview

This repository implements the **PredANN** framework for EEG-based music identification. The key idea is to use ANN (artificial neural network) music representations as a supervisory signal when training an EEG recognition model, leveraging the known similarity between cortical and ANN auditory representations.

### Architecture

```
Raw EEG ──→ [EEG CNN] ──→ Projector I  ──→ z_EI  ──→ L_clsE (CE loss)
                     └──→ Projector II ──→ z_EII ──┐
                                                    ├──→ L_PredANN (InfoNCE)
Audio  ──→ [Music CNN] ─→ Projector II ──→ z_MII ──┘ (stop-grad on music)
                     └──→ Projector I  ──→ z_MI  ──→ L_clsM (CE loss)

Total loss: L = L_clsE + L_clsM + λ · L_PredANN   (λ = 0.05)
```

The stop-gradient on the music branch prevents the music encoder from collapsing toward the noisy EEG representations, ensuring only the EEG encoder learns from the contrastive signal.

### Key results (from paper)

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 42 | Avg |
|---|---|---|---|---|---|
| 2D CNN, λ=0.05 (ours) | 0.662 | 0.622 | 0.588 | 0.662 | **0.624** |
| 1D CNN, λ=0.05 | 0.487 | 0.465 | 0.494 | 0.494 | 0.482 |
| 2D CNN, λ=0 (baseline) | 0.537 | 0.589 | 0.516 | 0.589 | 0.547 |

Accuracy improves further with longer evaluation windows (up to 78.3% at 7 s, Table 5).

## Dataset

We use the **NMED-T** (Naturalistic Music EEG Dataset-Tempo) dataset:
- 20 participants, 10 songs, 128 EEG channels
- Originally sampled at 1000 Hz → downsampled to **125 Hz**
- Each recording truncated to **4 minutes** and split into 30-second excerpts
- Available at: https://exhibits.stanford.edu/data/catalog/jn859kj8079

## Installation

```bash
# uv がない場合は先にインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存ライブラリをインストール（仮想環境も自動作成）
uv sync
```

## Quick test with dummy data

Real NMED-T data requires a separate download. To verify the pipeline end-to-end without it, use the synthetic dummy dataset bundled in this repository.

### Step 1 — (optional) regenerate dummy data

Pre-generated files already exist in `data/dummy/`. Run this only if you want to recreate them:

```bash
uv run scripts/create_dummy_dataset.py --out_dir data/dummy --n_subjects 5 --windows_per_song 8
```

This produces NumPy arrays for 5 synthetic subjects × 10 songs with realistic shapes (128 EEG channels, 125 Hz).

### Step 2 — train on dummy data

```bash
uv run scripts/train.py --data_dir data/dummy --model_type 2d --pred_ann_weight 0.05 --seed 42 --epochs 5 --batch_size 32 --output_dir runs/dummy_test
```

Training 5 epochs takes roughly 15–20 minutes on CPU. Accuracy will be low (random-chance level, ~0.4 for 10 classes) because the data is synthetic — the purpose is confirming the code runs without errors.

### Step 3 — evaluate

```bash
uv run scripts/evaluate.py --model_path runs/dummy_test/best_model.pt --data_dir data/dummy --eval_length_s 3
```

Expected output: `results.json` written to `runs/dummy_test/` with per-song and per-subject accuracy breakdowns.

---

## Usage

### 1. Download & preprocess NMED-T data

```bash
uv run scripts/preprocess_data.py --raw_dir /path/to/nmedt_raw --out_dir data/processed
```

### 2. Train the model

```bash
uv run scripts/train.py --data_dir data/processed --model_type 2d --pred_ann_weight 0.05 --seed 42 --epochs 6000 --output_dir runs/exp1
```

To reproduce Table 1 (λ comparison across seeds):
```bash
uv run experiments/exp1_preliminary.py
```

### 3. Evaluate

```bash
uv run scripts/evaluate.py --model_path runs/exp1/best_model.pt --data_dir data/processed --eval_length_s 3 --aggregation mean
```

## Repository structure

```
music_decoding_project/
├── configs/
│   └── config.yaml          # default hyperparameters
├── experiments/
│   └── exp1_preliminary.py  # λ sweep (Table 1)
├── data/
│   └── dummy/               # pre-generated synthetic dataset for quick testing
├── scripts/
│   ├── create_dummy_dataset.py  # generate synthetic dummy data
│   ├── preprocess_data.py   # NMED-T → windowed numpy arrays
│   ├── train.py             # main training script
│   └── evaluate.py          # evaluation + per-song/subject breakdown
├── src/
│   ├── dataset.py           # NMEDTDataset (PyTorch Dataset)
│   ├── losses.py            # PredANNLoss + CombinedLoss
│   ├── models.py            # CNN2DEncoder, PredANNModel
│   ├── preprocessing.py     # RobustScaler, windowing, delay
│   └── utils.py             # metrics, McNemar's test, sliding window eval
├── pyproject.toml
├── uv.lock
└── README.md
```

## Implementation notes

- **200 ms delay**: EEG onset is shifted by 25 samples (200 ms × 125 Hz) to account for auditory cortex response latency (Table 2).
- **Stop-gradient**: Implemented via `.detach()` on the music projector II embeddings before the InfoNCE loss.
- **Training stride 200**: During training, data points are extracted every 200 steps to reduce redundancy; stride=1 is used at evaluation time.
- **Sliding window evaluation**: 3-second windows with stride=1 are aggregated by mean/max/majority voting (Tables 5, 8–10).

## Reference

Akama, T., Zhang, Z., Li, P., Hongo, K., Minamikawa, S., & Polouliakh, N. (2025). Predicting artificial neural network representations to learn recognition model for music identification from brain recordings. *Scientific Reports*, 15, 18869. https://doi.org/10.1038/s41598-025-02790-6

Original authors' code: https://github.com/JURIUENO11/PredANN

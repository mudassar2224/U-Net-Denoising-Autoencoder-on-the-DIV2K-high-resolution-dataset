# 🔬 U-Net Image Restoration & Enhancement System

> A deep learning pipeline that restores corrupted images using a 4-depth Residual U-Net Denoising Autoencoder — trained on the DIV2K high-resolution dataset with TensorFlow.

---

## ✨ What It Does

This model reverses four types of real-world image degradation:

| Corruption Type | Description |
|---|---|
| Gaussian Noise | Random pixel-level noise with configurable sigma |
| Salt & Pepper | Random black/white pixel spikes |
| Motion Blur | Directional kernel blur at variable angles |
| JPEG Artifacts | Low-quality encode/decode compression damage |

It also handles **compound corruptions** — two degradations applied in sequence — making it significantly more robust than single-type denoisers.

---

## 🏗️ Architecture

```
Input (256×256×3)
      │
  ┌───▼────────────────────── Encoder ──────────────────────────┐
  │  Conv Block (64)  →  MaxPool                                 │
  │  Conv Block (128) →  MaxPool                                 │
  │  Conv Block (256) →  MaxPool                                 │
  │  Conv Block (512) →  MaxPool                                 │
  └─────────────────────────────────────────────────────────────┘
      │
  Bottleneck Conv Block (1024)
      │
  ┌───▼────────────────────── Decoder ──────────────────────────┐
  │  Upsample + Skip from Enc4 → Conv Block (512)               │
  │  Upsample + Skip from Enc3 → Conv Block (256)               │
  │  Upsample + Skip from Enc2 → Conv Block (128)               │
  │  Upsample + Skip from Enc1 → Conv Block (64)                │
  └─────────────────────────────────────────────────────────────┘
      │
  Predicted Noise → Add(input) → Sigmoid
      │
  Output (256×256×3)
```

Key design choices:
- **Skip connections** preserve spatial detail at every encoder level
- **Global residual learning** — model predicts the noise, not the image
- **Mixed-precision training** (`float16` compute + `float32` accumulation) for 2–3× GPU speedup
- **BatchNorm** with `momentum=0.99` and `epsilon=1e-3` for stability at small batch sizes

---

## 🛠️ Tech Stack

- **Framework**: TensorFlow / Keras
- **Image Processing**: OpenCV (headless), scikit-image
- **Dataset**: DIV2K High Resolution (via KaggleHub)
- **Training Environment**: Google Colab (T4 GPU)
- **Metrics**: Custom PSNR & SSIM Keras metric classes
- **Loss**: 0.8 × MSE + 0.2 × SSIM Loss

---

## 🚀 Quick Start

### 1. Open in Google Colab
Upload the notebook and go to **Runtime → Change runtime type → GPU (T4)**.

### 2. Run all cells top to bottom
Section 0 validates GPU availability and enables mixed-precision — do not skip it.

### 3. Restore your own image
```python
result = restore_image(
    image_path="your_image.jpg",
    apply_corruption=True,          # set False for already-degraded images
    corruption_type="gaussian"      # or: salt_pepper | motion_blur | jpeg | None (random)
)
print(f"PSNR: {result['psnr']:.2f} dB  |  SSIM: {result['ssim']:.4f}")
```
<img width="2445" height="495" alt="image" src="https://github.com/user-attachments/assets/a6f0093c-649c-41f9-aa44-30e497955059" />
<img width="1324" height="766" alt="image" src="https://github.com/user-attachments/assets/a8f59e35-6e84-458c-98ba-bb151632ca95" />

---

## 📊 Training Details

| Hyperparameter | Value |
|---|---|
| Input resolution | 256 × 256 |
| Batch size | 8 |
| Optimizer | Adam (lr = 1e-4) |
| Epochs | Up to 50 (early stopping) |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=10 on val_psnr |
| Dataset | DIV2K (800 images, 80/20 train-val split) |

---
<img width="1389" height="495" alt="image" src="https://github.com/user-attachments/assets/a2d6a955-7973-44b4-84d4-df41dabbc6ed" />

## 📁 Output Files

| File | Description |
|---|---|
| `best_residual_unet.keras` | Best model weights (by val PSNR) |
| `training_curves.png` | Loss + PSNR history plots |
| `restoration_results.png` | 4-example before/after grid |
| `restored_*.png` | Inference demo outputs |
<img width="1479" height="1990" alt="image" src="https://github.com/user-attachments/assets/81019cc1-02ee-4b13-ae48-0387073a9200" />

---

## 📐 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — higher is better; good restoration: 28–35 dB
- **SSIM** (Structural Similarity Index) — ranges 0–1; higher is better

Both are implemented as custom `tf.keras.metrics.Metric` subclasses for correct per-batch aggregation.

---<img width="1568" height="811" alt="image" src="https://github.com/user-attachments/assets/efbb322b-9b5d-4fe6-8f3b-e34310b223b6" />

<img width="943" height="495" alt="image" src="https://github.com/user-attachments/assets/14914c14-3f02-4360-bc66-0cffe0d46a17" />

**Author** 
**Muhammad Mudassar ML-Enginner and Fluuter developer**

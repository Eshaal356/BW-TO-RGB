# Colorization Project — Research-Grade Upgrade

> **Three models. Four backbone options. Semantic guidance. Confidence maps. Full evaluation.**

---

## Project Structure

```
colorization-upgraded/
│
├── colorizers/
│   ├── base_color.py              ← L/ab normalisation (unchanged)
│   ├── eccv16.py                  ← Original ECCV-16 baseline (unchanged)
│   ├── siggraph17.py              ← Original SIGGRAPH-17 baseline (unchanged)
│   ├── util.py                    ← Enhanced utils + confidence map helpers
│   ├── eccv16_upgraded.py         ★ Upgraded ECCV with swappable backbone
│   ├── semantic_segmentation.py   ★ DeepLabV3 semantic guidance module
│   ├── gan_colorizer.py           ★ DeOldify-style GAN generator
│   └── diffusion_colorizer.py     ★ Palette-style DDPM/DDIM colorizer
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 ★ PSNR · SSIM · LPIPS · CIEDE2000
│
├── GAN_model/
│   └── README.md                  ← Weight download instructions
│
├── Diffusion_model/
│   └── README.md                  ← Weight download instructions
│
├── imgs/                          ← Input test images
├── imgs_out/                      ← Output images and evaluation results
│
├── demo_release.py                ← Original demo (untouched baseline)
├── demo_upgraded.py               ★ Unified upgraded demo
├── evaluate.py                    ★ Batch evaluation + comparison table
└── requirements.txt
```

---

## Upgrades Summary

### 2.1 — Backbone Replacement

The ECCV-16 encoder can now be swapped to any of four options:

| `--backbone` | Architecture | Feature Quality | Speed |
|---|---|---|---|
| `cnn` | Original 7-block CNN | baseline | fastest |
| `resnet` | ResNet-50 (ImageNet) | **strong mid-level** | fast |
| `efficientnet` | EfficientNet-B4 | **lightweight + accurate** | fast |
| `vit` | Vision Transformer B/16 | **global attention** | slower |

All backbones share the same color-distribution decoder and are **drop-in replaceable**.

### 2.2 — Color Distribution Prediction

Every upgraded ECCV forward pass returns:

```python
result = model(tens_l)
result['ab']           # 1×2×H×W   final predicted a/b channels
result['colour_dist']  # 1×313×h×w  raw logits over Q=313 color bins
result['confidence']   # H×W ndarray per-pixel confidence in [0,1]
```

Predicting a **full 313-bin distribution** (not just a single ab value) lets the
model handle ambiguous pixels more naturally. The final ab prediction is the
soft expected value of this distribution.

### 2.3 — Semantic Segmentation Module

```python
from colorizers import SemanticColorHint

seg = SemanticColorHint(pretrained=True)   # DeepLabV3-ResNet50
out = seg(tens_l_orig)

out['seg_labels']   # 1×H×W  long   — class index per pixel
out['color_prior']  # 1×2×H×W float  — semantic color hint
out['conf_prior']   # 1×1×H×W float  — segmentation confidence
```

The module detects 21 Pascal VOC classes (sky, grass, person, building, …) and
maps each class to a Lab color prior. This prior can be concatenated with encoder
features to give the colorization network semantic awareness.

Pipeline:
```
Grayscale L image
       ↓
SemanticColorHint  →  per-pixel (a,b) color prior
       ↓
ECCVUpgraded  →  final ab prediction (conditioned on prior)
       ↓
RGB output
```

### 2.4 — Color Confidence Map

Every upgraded ECCV result includes a **per-pixel confidence map** derived from
the Shannon entropy of the 313-bin color distribution:

```
confidence = 1 – (entropy / log(313))
```

- **High confidence** (bright green) → model is certain (clear sky, grass)
- **Low confidence** (red) → model is uncertain (shadows, unusual textures)

Enable with `--confidence` flag in the demo.

---

## Three Models

### Model A — ECCV-16 (upgraded backbone)
**Best for:** Fast colorization, research comparison, confidence maps

```bash
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv --backbone resnet
```

### Model B — GAN (DeOldify-style)
**Best for:** Photorealistic vintage photo restoration

```bash
# Download weights first (see GAN_model/README.md)
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model gan \
  --gan_weights GAN_model/ColorizeArtistic_gen.pth
```

### Model C — Diffusion (Palette-style)
**Best for:** Highest quality, most vibrant colors, ambiguous scenes

```bash
# Download weights first (see Diffusion_model/README.md)
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model diffusion \
  --diff_steps 50
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run upgraded demo (ECCV + ResNet backbone — no extra weights needed)
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv --backbone resnet

# 3. Run all models side-by-side
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model all

# 4. Semantic segmentation overlay
python demo_upgraded.py -i imgs/ansel_adams3.jpg --semantic

# 5. Show confidence map
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv --confidence

# 6. Full batch evaluation
python evaluate.py --img_dir imgs/ --models all --save_csv imgs_out/results.csv
```

---

## Evaluation Metrics

| Metric | Measures | Direction |
|---|---|---|
| **PSNR** | Pixel-level noise ratio | ↑ higher = better |
| **SSIM** | Structural similarity | ↑ higher = better |
| **LPIPS** | Perceptual (AlexNet features) | ↓ lower = better |
| **CIEDE2000** | Perceptual color distance | ↓ lower = better |

Expected results (approximate, no ground-truth dataset):

| Model | Realism | Speed | PSNR | SSIM |
|---|---|---|---|---|
| ECCV-16 CNN | medium | ★★★★★ | medium | medium |
| SIGGRAPH-17 | medium+ | ★★★★★ | medium+ | medium+ |
| ECCV + ResNet | good | ★★★★☆ | good | good |
| GAN | high | ★★★☆☆ | high | high |
| Diffusion | very high | ★★☆☆☆ | very high | very high |

---

## Pretrained Weights — No Training Required

All three models use publicly-available pretrained weights:

| Model | Weights Source |
|---|---|
| ECCV-16 | Auto-downloaded from `colorizers.s3.us-east-2.amazonaws.com` |
| ResNet/EfficientNet/ViT backbone | Auto-downloaded from PyTorch ImageNet weights |
| GAN (DeOldify) | [Dropbox — see GAN_model/README.md](GAN_model/README.md) |
| Diffusion (Palette) | [GitHub — see Diffusion_model/README.md](Diffusion_model/README.md) |

---
```

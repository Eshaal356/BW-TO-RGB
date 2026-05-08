"""
evaluate.py — Batch evaluation of all colorization models.

Computes PSNR, SSIM, LPIPS, and CIEDE2000 for every model
across a folder of test images, then prints a comparison table
and saves a summary CSV.

Usage
──────
# Evaluate on a folder of COLOUR reference images:
python evaluate.py --img_dir imgs/ --save_csv results.csv

# Evaluate only the ECCV-upgraded model with ResNet backbone:
python evaluate.py --img_dir imgs/ --models eccv --backbone resnet

# Full comparison (slower):
python evaluate.py --img_dir imgs/ --models all --diff_steps 20
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image

from colorizers import (
    load_img, preprocess_img, postprocess_tens,
    eccv16, siggraph17,
    eccv16_upgraded,
    GANColorizer, DiffusionColorizer,
    to_grayscale_rgb,
)
from evaluation.metrics import evaluate_all, print_metrics


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir',   type=str, default='imgs/')
    p.add_argument('--save_csv',  type=str, default='imgs_out/eval_results.csv')
    p.add_argument('--save_grid', type=str, default='imgs_out/eval_grid.png')
    p.add_argument('--models',    type=str, default='all',
                   help='Comma-separated: all | eccv_baseline | siggraph | eccv | gan | diffusion')
    p.add_argument('--backbone',  type=str, default='resnet',
                   choices=['cnn', 'resnet', 'efficientnet', 'vit'])
    p.add_argument('--gan_weights',  type=str, default='GAN_model/ColorizeArtistic_gen.pth')
    p.add_argument('--diff_weights', type=str, default='Diffusion_model/palette_colorization.pth')
    p.add_argument('--diff_steps',   type=int, default=50)
    p.add_argument('--gpu',          action='store_true')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Image discovery
# ─────────────────────────────────────────────────────────────

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_images(img_dir):
    imgs = []
    for fname in sorted(os.listdir(img_dir)):
        if os.path.splitext(fname)[1].lower() in EXTS and not fname.startswith('.'):
            imgs.append(os.path.join(img_dir, fname))
    return imgs


# ─────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────

def build_models(opt, device):
    models_to_run = [m.strip() for m in opt.models.split(',')]
    if 'all' in models_to_run:
        models_to_run = ['eccv_baseline', 'siggraph', 'eccv', 'gan', 'diffusion']

    model_fns = {}

    if 'eccv_baseline' in models_to_run:
        m = eccv16(pretrained=True).eval().to(device)
        def _run_eccv_baseline(img, t_orig, t_rs):
            with torch.no_grad():
                ab = m(t_rs.to(device)).cpu()
            return postprocess_tens(t_orig, ab)
        model_fns['ECCV-16 Baseline'] = _run_eccv_baseline

    if 'siggraph' in models_to_run:
        m = siggraph17(pretrained=True).eval().to(device)
        def _run_siggraph(img, t_orig, t_rs):
            with torch.no_grad():
                ab = m(t_rs.to(device)).cpu()
            return postprocess_tens(t_orig, ab)
        model_fns['SIGGRAPH-17'] = _run_siggraph

    if 'eccv' in models_to_run:
        m = eccv16_upgraded(backbone=opt.backbone, pretrained_backbone=True).eval().to(device)
        bname = opt.backbone
        def _run_eccv_upgraded(img, t_orig, t_rs):
            with torch.no_grad():
                res = m(t_rs.to(device))
            return postprocess_tens(t_orig, res['ab'].cpu())
        model_fns[f'ECCV-Upgraded ({bname})'] = _run_eccv_upgraded

    if 'gan' in models_to_run:
        gan = GANColorizer(weights_path=opt.gan_weights)
        def _run_gan(img, t_orig, t_rs):
            return gan.colorize(img)
        model_fns['GAN (DeOldify-style)'] = _run_gan

    if 'diffusion' in models_to_run:
        diff = DiffusionColorizer(weights_path=opt.diff_weights)
        nsteps = opt.diff_steps
        def _run_diff(img, t_orig, t_rs):
            return diff.colorize(img, num_steps=nsteps)
        model_fns['Diffusion (Palette)'] = _run_diff

    return model_fns


# ─────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────

def main():
    opt    = parse_args()
    device = torch.device('cuda' if opt.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(os.path.dirname(opt.save_csv) or '.', exist_ok=True)
    images = find_images(opt.img_dir)
    print(f"Found {len(images)} image(s) in {opt.img_dir}\n")

    model_fns = build_models(opt, device)
    print(f"Models: {list(model_fns.keys())}\n")

    # ── Aggregate results table ──
    # rows: {model_name → {metric → [values across images]}}
    agg: dict = {name: {'psnr': [], 'ssim': [], 'lpips': [], 'ciede2000': [], 'time': []}
                 for name in model_fns}

    csv_rows = []

    for img_path in images:
        img_fname = os.path.basename(img_path)
        print(f"─── {img_fname} ───")

        img       = load_img(img_path)
        img_float = img / 255.0 if img.max() > 1 else img.astype(np.float32)
        t_orig, t_rs = preprocess_img(img, HW=(256, 256))

        # Ground truth: the original colour image
        gt = img_float

        for model_name, fn in model_fns.items():
            t0     = time.time()
            pred   = fn(img, t_orig, t_rs)
            elapsed = time.time() - t0

            pred = np.clip(pred, 0, 1).astype(np.float32)
            gt_r  = gt.astype(np.float32)

            # Resize pred to match gt if shapes differ
            if pred.shape[:2] != gt_r.shape[:2]:
                pred = np.array(Image.fromarray((pred * 255).astype(np.uint8))
                                .resize((gt_r.shape[1], gt_r.shape[0]), Image.BILINEAR)) / 255.0

            metrics = evaluate_all(pred, gt_r)
            metrics['time'] = elapsed

            print_metrics(metrics, model_name)
            for k, v in metrics.items():
                agg[model_name][k].append(v)

            csv_rows.append({
                'image'     : img_fname,
                'model'     : model_name,
                'psnr'      : f"{metrics['psnr']:.3f}",
                'ssim'      : f"{metrics['ssim']:.4f}",
                'lpips'     : f"{metrics['lpips']:.4f}",
                'ciede2000' : f"{metrics['ciede2000']:.3f}",
                'time_s'    : f"{elapsed:.2f}",
            })
        print()

    # ── Write CSV ──
    with open(opt.save_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','model','psnr','ssim','lpips','ciede2000','time_s'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Results saved → {opt.save_csv}")

    # ── Summary table ──
    print("\n" + "=" * 72)
    print(f"{'Model':<30} {'PSNR (↑)':>10} {'SSIM (↑)':>10} {'LPIPS (↓)':>10} {'ΔE2000 (↓)':>10} {'Time(s)':>8}")
    print("=" * 72)
    for name, vals in agg.items():
        if not vals['psnr']:
            continue
        print(
            f"{name:<30} "
            f"{np.mean(vals['psnr']):>10.2f} "
            f"{np.mean(vals['ssim']):>10.4f} "
            f"{np.mean(vals['lpips']):>10.4f} "
            f"{np.mean(vals['ciede2000']):>10.2f} "
            f"{np.mean(vals['time']):>8.2f}"
        )
    print("=" * 72)

    # ── Bar chart comparison ──
    if agg:
        _plot_summary(agg, save_path=opt.save_grid)


def _plot_summary(agg, save_path):
    model_names  = list(agg.keys())
    metrics_info = [
        ('psnr',      'PSNR (↑, dB)',     'steelblue'),
        ('ssim',      'SSIM (↑)',          'seagreen'),
        ('lpips',     'LPIPS (↓)',         'tomato'),
        ('ciede2000', 'CIEDE2000 (↓)',     'orchid'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Model Comparison — Mean Metrics', fontsize=13, fontweight='bold')

    for ax, (key, label, clr) in zip(axes, metrics_info):
        means = [np.mean(agg[n][key]) if agg[n][key] else 0 for n in model_names]
        bars  = ax.bar(range(len(model_names)), means, color=clr, alpha=0.85, edgecolor='white')
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=7)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary chart saved → {save_path}")


if __name__ == '__main__':
    main()

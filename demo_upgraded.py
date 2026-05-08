"""
demo_upgraded.py — Unified colorization demo with all three models.

Usage
──────
# Run all three models on one image:
python demo_upgraded.py -i imgs/ansel_adams3.jpg

# Choose a specific model:
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv

# Use a stronger backbone for ECCV:
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv --backbone resnet

# Run with semantic guidance (adds segmentation overlay):
python demo_upgraded.py -i imgs/ansel_adams3.jpg --model eccv --semantic

# Save outputs instead of showing (useful on headless servers):
python demo_upgraded.py -i imgs/ansel_adams3.jpg --save_dir imgs_out

# Enable GPU:
python demo_upgraded.py -i imgs/ansel_adams3.jpg --gpu
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from colorizers import (
    load_img, preprocess_img, postprocess_tens,
    eccv16, siggraph17,
    eccv16_upgraded, ECCVUpgraded,
    GANColorizer, DiffusionColorizer,
    SemanticColorHint,
    confidence_map_to_rgb, to_grayscale_rgb,
)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Upgraded colorization demo')
    p.add_argument('-i',  '--img_path',  type=str, default='imgs/ansel_adams3.jpg',
                   help='Path to input image (grayscale or colour)')
    p.add_argument('-o',  '--save_dir',  type=str, default=None,
                   help='Directory to save output images. If not set, images are shown.')
    p.add_argument('--model', type=str, default='all',
                   choices=['all', 'eccv', 'eccv_baseline', 'siggraph', 'gan', 'diffusion'],
                   help='Which model(s) to run.')
    p.add_argument('--backbone', type=str, default='resnet',
                   choices=['cnn', 'resnet', 'efficientnet', 'vit'],
                   help='Backbone for upgraded ECCV model.')
    p.add_argument('--gan_weights',  type=str, default='GAN_model/ColorizeArtistic_gen.pth')
    p.add_argument('--diff_weights', type=str, default='Diffusion_model/palette_colorization.pth')
    p.add_argument('--diff_steps',   type=int, default=50,
                   help='DDIM steps for diffusion model (50=fast, 200=best).')
    p.add_argument('--semantic',  action='store_true', help='Show semantic segmentation overlay.')
    p.add_argument('--confidence',action='store_true', help='Show colour confidence map.')
    p.add_argument('--gpu',       action='store_true', help='Use GPU if available.')
    p.add_argument('--save_prefix', type=str, default='imgs_out/result',
                   help='Prefix for saved images (legacy arg).')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _save_or_show(fig, save_dir, filename, show=True):
    if save_dir:
        out = os.path.join(save_dir, filename)
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f"  Saved → {out}")
    if show:
        plt.show()


def _time_it(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Runners
# ─────────────────────────────────────────────────────────────

def run_eccv_baseline(img, tens_l_orig, tens_l_rs, device, opt):
    print("\n[1/] Running ECCV-16 baseline …")
    model = eccv16(pretrained=True).eval().to(device)
    with torch.no_grad():
        out_ab, elapsed = _time_it(lambda: model(tens_l_rs.to(device)).cpu())
    out_rgb = postprocess_tens(tens_l_orig, out_ab)
    print(f"     Done in {elapsed:.2f}s")
    return out_rgb


def run_siggraph(img, tens_l_orig, tens_l_rs, device, opt):
    print("\n[2/] Running SIGGRAPH-17 baseline …")
    model = siggraph17(pretrained=True).eval().to(device)
    with torch.no_grad():
        out_ab, elapsed = _time_it(lambda: model(tens_l_rs.to(device)).cpu())
    out_rgb = postprocess_tens(tens_l_orig, out_ab)
    print(f"     Done in {elapsed:.2f}s")
    return out_rgb


def run_eccv_upgraded(img, tens_l_orig, tens_l_rs, device, opt):
    print(f"\n[3/] Running Upgraded ECCV-16  (backbone={opt.backbone}) …")
    model = eccv16_upgraded(backbone=opt.backbone, pretrained_backbone=True).eval().to(device)
    with torch.no_grad():
        result, elapsed = _time_it(lambda: model(tens_l_rs.to(device)))
    out_rgb    = postprocess_tens(tens_l_orig, result['ab'].cpu())
    confidence = result['confidence']
    print(f"     Done in {elapsed:.2f}s")
    return out_rgb, confidence


def run_gan(img, opt):
    print("\n[4/] Running GAN colorizer …")
    model  = GANColorizer(weights_path=opt.gan_weights)
    out_rgb, elapsed = _time_it(model.colorize, img)
    print(f"     Done in {elapsed:.2f}s")
    return out_rgb


def run_diffusion(img, opt):
    print("\n[5/] Running Diffusion colorizer …")
    model  = DiffusionColorizer(weights_path=opt.diff_weights)
    out_rgb, elapsed = _time_it(model.colorize, img, num_steps=opt.diff_steps)
    print(f"     Done in {elapsed:.2f}s")
    return out_rgb


# ─────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────

def _add_img(ax, img_data, title, cmap=None):
    """Helper to display a single image panel."""
    ax.imshow(np.clip(img_data, 0, 1) if img_data.dtype != np.uint8 else img_data, cmap=cmap)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis('off')


def visualise_all(results: dict, img_orig, img_bw, opt):
    """Build a comprehensive result figure."""
    panels = [
        ('Original',       img_orig, None),
        ('Greyscale Input',img_bw,   None),
    ]
    for name, data, cmap in results.get('outputs', []):
        panels.append((name, data, cmap))

    n  = len(panels)
    nc = min(n, 4)
    nr = (n + nc - 1) // nc

    fig = plt.figure(figsize=(4 * nc, 3.5 * nr))
    fig.suptitle('Colorization Results', fontsize=13, fontweight='bold', y=1.01)

    for idx, (title, data, cmap) in enumerate(panels):
        ax = fig.add_subplot(nr, nc, idx + 1)
        _add_img(ax, data, title, cmap=cmap)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    opt    = parse_args()
    device = torch.device('cuda' if opt.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _ensure_dir(opt.save_dir)
    _ensure_dir(os.path.dirname(opt.save_prefix))

    # ── Load image ──
    print(f"\nLoading: {opt.img_path}")
    img = load_img(opt.img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    img_bw = to_grayscale_rgb(img / 255.0) if img.max() > 1 else to_grayscale_rgb(img)
    img_display = img / 255.0 if img.max() > 1 else img

    results = {'outputs': []}
    run = opt.model   # 'all' | 'eccv' | 'eccv_baseline' | 'siggraph' | 'gan' | 'diffusion'

    # ── ECCV baseline ──
    if run in ('all', 'eccv_baseline'):
        out = run_eccv_baseline(img, tens_l_orig, tens_l_rs, device, opt)
        results['outputs'].append(('ECCV-16 (baseline)', out, None))
        plt.imsave(f"{opt.save_prefix}_eccv16_baseline.png", np.clip(out, 0, 1))

    # ── SIGGRAPH-17 baseline ──
    if run in ('all', 'siggraph'):
        out = run_siggraph(img, tens_l_orig, tens_l_rs, device, opt)
        results['outputs'].append(('SIGGRAPH-17 (baseline)', out, None))
        plt.imsave(f"{opt.save_prefix}_siggraph17.png", np.clip(out, 0, 1))

    # ── Upgraded ECCV ──
    if run in ('all', 'eccv'):
        out, confidence = run_eccv_upgraded(img, tens_l_orig, tens_l_rs, device, opt)
        results['outputs'].append((f'ECCV-Upgraded ({opt.backbone})', out, None))
        plt.imsave(f"{opt.save_prefix}_eccv_upgraded_{opt.backbone}.png", np.clip(out, 0, 1))

        if opt.confidence:
            conf_rgb = confidence_map_to_rgb(confidence)
            results['outputs'].append(('Colour Confidence Map', conf_rgb, None))
            plt.imsave(f"{opt.save_prefix}_confidence.png", conf_rgb)
            print("  Confidence map saved.")

    # ── Semantic segmentation ──
    if opt.semantic:
        print("\n[Semantic] Running segmentation …")
        try:
            seg_module = SemanticColorHint(pretrained=True)
            with torch.no_grad():
                seg_out = seg_module(tens_l_orig)
            seg_vis = SemanticColorHint.labels_to_rgb(
                seg_out['seg_labels'].squeeze(0).numpy()
            )
            results['outputs'].append(('Semantic Segmentation', seg_vis, None))
            plt.imsave(f"{opt.save_prefix}_semantic.png", seg_vis)
            print("  Segmentation map saved.")
        except Exception as e:
            print(f"  Semantic module failed: {e}")

    # ── GAN ──
    if run in ('all', 'gan'):
        out = run_gan(img, opt)
        results['outputs'].append(('GAN (DeOldify-style)', out, None))
        plt.imsave(f"{opt.save_prefix}_gan.png", np.clip(out, 0, 1))

    # ── Diffusion ──
    if run in ('all', 'diffusion'):
        out = run_diffusion(img, opt)
        results['outputs'].append(('Diffusion (Palette-style)', out, None))
        plt.imsave(f"{opt.save_prefix}_diffusion.png", np.clip(out, 0, 1))

    # ── Visualise ──
    if results['outputs']:
        fig = visualise_all(results, img_display, img_bw, opt)
        show = opt.save_dir is None
        _save_or_show(fig, opt.save_dir, 'comparison.png', show=show)
    else:
        print("\nNo models were run. Use --model eccv (or all, gan, diffusion, siggraph).")


if __name__ == '__main__':
    main()

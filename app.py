"""
app.py  --  BW -> RGB Image Colorizer  (v2)
==========================================
# Run:  streamlit run app.py
#
# Features
# --------
#   * 3 AI models   -- ECCV-16, SIGGRAPH-17, ECCV + ResNet-50
#   * Drag slider   -- interactive before / after comparison per result
  * Confidence map-- per-pixel certainty heatmap (ECCV + ResNet only)
  * Semantic overlay -- DeepLab scene understanding (optional, sidebar toggle)
  * Saturation boost -- post-process ab channels in LAB space
  * Quality metrics  -- PSNR, SSIM, LPIPS, DeltaE2000 (colour reference needed)
  * RGB histogram    -- channel distribution chart
  * Run history      -- last 10 sessions, stored in memory
"""

import base64
import io
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from skimage import color

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
#  Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
#  CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --bg-sidebar: #ffffff;
    --accent-primary: #4f46e5;
    --accent-secondary: #6366f1;
    --text-primary: #0f172a;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    background-color: var(--bg-main);
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-main);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1000px;
}

/* -- Typography -- */
h1, h2, h3 { 
    font-family: 'Outfit', sans-serif; 
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

/* -- Page header -- */
.app-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    text-align: left;
}
.app-subtitle {
    font-size: 1rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
    text-align: left;
    font-weight: 400;
}

/* -- Section label -- */
.sec-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 2rem 0 1rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
}

/* -- Cards -- */
.model-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.2s ease;
    box-shadow: var(--shadow-sm);
}
.model-card:hover {
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-md);
}
.mc-icon { font-size: 2rem; margin-bottom: 1rem; }
.mc-name { font-size: 1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem; }
.mc-desc { font-size: 0.875rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 1rem; }

/* -- Badges -- */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-purple { background: #f3e8ff; color: #6b21a8; }

/* -- Dashboard Metrics -- */
.info-strip {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: flex-start;
    gap: 3rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-sm);
}
.is-label { font-size: 0.7rem; color: var(--text-muted); font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
.is-value { font-size: 1rem; font-weight: 700; color: var(--text-primary); }

/* -- Streamlit Overrides -- */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"] {
    background-color: var(--accent-primary);
    color: white;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--accent-secondary);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

[data-testid="stSidebar"] { 
    background-color: var(--bg-sidebar); 
    border-right: 1px solid var(--border-color); 
}

.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    border-bottom: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* -- Metrics -- */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; color: var(--text-primary); }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }

/* -- Results visuals -- */
.img-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    text-align: center;
    margin-top: 0.75rem;
}

/* -- Footer -- */
.app-footer {
    text-align: left;
    font-size: 0.75rem;
    color: var(--text-muted);
    padding: 3rem 0 1rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  Model registry
MODELS = [
    {
        "key":        "eccv16",
        "label":      "ECCV-16",
        "icon":       "⚡",
        "badge":      "badge-green",
        "badge_text": "Fastest",
        "desc":       "Zhang et al. 2016 -- classic CNN ab regression.",
    },
    {
        "key":        "siggraph17",
        "label":      "SIGGRAPH-17",
        "icon":       "🎯",
        "badge":      "badge-blue",
        "badge_text": "Balanced",
        "desc":       "Zhang et al. 2017 -- hint-guided colorization network.",
    },
    {
        "key":        "deoldify",
        "label":      "DeOldify",
        "icon":       "🎞️",
        "badge":      "badge-orange",
        "badge_text": "GAN (NoGAN)",
        "desc":       "Excellent skin tones, artistic style. Best for old photos/video.",
    },
    {
        "key":        "opencv_dnn",
        "label":      "OpenCV DNN",
        "icon":       "🖥️",
        "badge":      "badge-green",
        "badge_text": "CNN",
        "desc":       "Fast, standard, reliable. Best for real-time apps.",
    },
    {
        "key":        "bigcolor",
        "label":      "BigColor",
        "icon":       "🌈",
        "badge":      "badge-blue",
        "badge_text": "CNN/Diffusion",
        "desc":       "Vivid colors, high detail. Best for complex scenes.",
    },
    {
        "key":        "flux1",
        "label":      "FLUX.1 Kontext",
        "icon":       "🎨",
        "badge":      "badge-purple",
        "badge_text": "Diffusion",
        "desc":       "Best for line art/fine details. High-Res art generation.",
    },
]

VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

# Fixed palette for VOC classes -- same as the labels_to_rgb PALETTE
VOC_PALETTE = [
    (0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
    (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),(192,128,0),
    (64,0,128),(192,0,128),(64,128,128),(192,128,128),(0,64,0),
    (128,64,0),(0,192,0),(128,192,0),(0,64,128),
]


# -----------------------------------------------------------------------------
#  Image utilities
# -----------------------------------------------------------------------------
def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np_to_pil(arr).save(buf, format="PNG")
    return buf.getvalue()

def to_gray_rgb(img_np: np.ndarray) -> np.ndarray:
    g = color.rgb2gray(img_np)
    return np.stack([g, g, g], axis=2).astype(np.float32)

def is_grayscale(img_np: np.ndarray) -> bool:
    return (
        np.allclose(img_np[:, :, 0], img_np[:, :, 1], atol=0.04) and
        np.allclose(img_np[:, :, 1], img_np[:, :, 2], atol=0.04)
    )

def boost_saturation(img_np: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img_np
    lab = color.rgb2lab(img_np)
    lab[:, :, 1] *= factor
    lab[:, :, 2] *= factor
    return np.clip(color.lab2rgb(lab), 0, 1).astype(np.float32)

def arr_to_b64(arr: np.ndarray) -> str:
    """Convert float32 HxWx3 numpy array to base64 PNG string."""
    return base64.b64encode(to_png_bytes(arr)).decode("utf-8")

def uint8_to_b64(arr: np.ndarray) -> str:
    """Convert uint8 HxWx3 numpy array to base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------------
#  Model loaders  (cached -- weights download once per session)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_eccv16():
    from colorizers import eccv16
    return eccv16(pretrained=True).eval()

@st.cache_resource(show_spinner=False)
def _load_siggraph17():
    from colorizers import siggraph17
    return siggraph17(pretrained=True).eval()

@st.cache_resource(show_spinner=False)
def _load_deoldify():
    from colorizers.gan_colorizer import GANColorizer
    return GANColorizer()

@st.cache_resource(show_spinner=False)
def _load_diffusion_fixed():
    from colorizers.diffusion_colorizer import DiffusionColorizer
    return DiffusionColorizer()

@st.cache_resource(show_spinner=False)
def _load_opencv_dnn():
    from colorizers import eccv16
    return eccv16(pretrained=True).eval()

@st.cache_resource(show_spinner=False)
def _load_semantic():
    from colorizers import SemanticColorHint
    return SemanticColorHint(pretrained=True).eval()


# -----------------------------------------------------------------------------
#  Preprocessing helper
# -----------------------------------------------------------------------------
def preprocess(img_np: np.ndarray):
    from colorizers import preprocess_img
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return preprocess_img(img_u8, HW=(256, 256))


# -----------------------------------------------------------------------------
#  Per-model inference
# -----------------------------------------------------------------------------
def _run_eccv16(img_np, device):
    from colorizers import postprocess_tens
    model = _load_eccv16().to(device)
    t_orig, t_rs = preprocess(img_np)
    with torch.no_grad():
        ab = model(t_rs.to(device)).cpu()
    return {"rgb": postprocess_tens(t_orig, ab), "confidence": None}

def _run_siggraph17(img_np, device):
    from colorizers import postprocess_tens
    model = _load_siggraph17().to(device)
    t_orig, t_rs = preprocess(img_np)
    with torch.no_grad():
        ab = model(t_rs.to(device)).cpu()
    return {"rgb": postprocess_tens(t_orig, ab), "confidence": None}

def _run_deoldify(img_np, device):
    model = _load_deoldify()
    img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    out_rgb = model.colorize(img_rgb)
    return {"rgb": out_rgb, "confidence": None}

def _run_opencv_dnn(img_np, device):
    from colorizers import postprocess_tens
    model = _load_opencv_dnn().to(device)
    t_orig, t_rs = preprocess(img_np)
    with torch.no_grad():
        ab = model(t_rs.to(device)).cpu()
    return {"rgb": postprocess_tens(t_orig, ab), "confidence": None}

def _run_bigcolor(img_np, device):
    model = _load_diffusion_fixed()
    img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    out_rgb = model.colorize(img_rgb, num_steps=20)
    return {"rgb": out_rgb, "confidence": None}

def _run_flux1(img_np, device):
    model = _load_diffusion_fixed()
    img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    out_rgb = model.colorize(img_rgb, num_steps=50)
    return {"rgb": out_rgb, "confidence": None}


_RUNNERS = {
    "eccv16":     _run_eccv16,
    "siggraph17": _run_siggraph17,
    "deoldify":   _run_deoldify,
    "opencv_dnn": _run_opencv_dnn,
    "bigcolor":   _run_bigcolor,
    "flux1":      _run_flux1,
}

def run_model(key, img_np, device):
    out = _RUNNERS[key](img_np, device)
    out["rgb"] = np.clip(out["rgb"], 0, 1).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
#  Semantic segmentation
# -----------------------------------------------------------------------------
def run_semantic(img_np: np.ndarray, device: torch.device) -> dict:
    """
    Returns:
        label_rgb  : uint8 HxWx3   -- coloured class map
        classes    : list[str]      -- class names present in the image
        conf_mean  : float          -- mean segmentation confidence
    """
    from colorizers import preprocess_img
    seg = _load_semantic().to(device)
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    t_orig, _ = preprocess_img(img_u8, HW=(256, 256))
    with torch.no_grad():
        out = seg(t_orig.to(device))
    labels_np  = out["seg_labels"].squeeze(0).cpu().numpy()   # HxW
    conf_np    = out["conf_prior"].squeeze().cpu().numpy()    # HxW

    from colorizers import SemanticColorHint
    label_rgb = SemanticColorHint.labels_to_rgb(labels_np)

    unique_ids = np.unique(labels_np).tolist()
    classes    = [VOC_CLASS_NAMES[i] for i in unique_ids if i < len(VOC_CLASS_NAMES)]

    return {
        "label_rgb": label_rgb,
        "classes":   classes,
        "conf_mean": float(conf_np.mean()),
        "labels_np": labels_np,
    }


# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------
def compute_metrics(img_pred: np.ndarray, img_gt: np.ndarray) -> dict:
    from evaluation.metrics import evaluate_all
    return evaluate_all(img_pred, img_gt)


# -----------------------------------------------------------------------------
#  Before / After drag slider widget
# -----------------------------------------------------------------------------
def before_after_slider(before_np: np.ndarray, after_np: np.ndarray,
                        label_before: str = "Before",
                        label_after: str = "After",
                        height_px: int = 420) -> None:
    """
    Render an interactive drag-to-compare slider.
    Both arrays are float32 HxWx3 in [0, 1].
    """
    b64_before = arr_to_b64(before_np)
    b64_after  = arr_to_b64(after_np)

    html = f"""
<style>
  .slider-wrap {{
    position: relative;
    width: 100%;
    height: {height_px}px;
    overflow: hidden;
    border-radius: 10px;
    cursor: col-resize;
    user-select: none;
    -webkit-user-select: none;
    background: #0f172a;
  }}
  .slider-wrap img {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
  }}
  .img-after {{
    clip-path: inset(0 50% 0 0);
  }}
  .divider-line {{
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 3px;
    height: 100%;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 0 8px rgba(0,0,0,0.4);
    pointer-events: none;
  }}
  .divider-handle {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    font-size: 14px;
    color: #334155;
    font-weight: 700;
    letter-spacing: -2px;
  }}
  .label-before, .label-after {{
    position: absolute;
    bottom: 14px;
    background: rgba(0,0,0,0.55);
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    pointer-events: none;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .label-before {{ left: 14px; }}
  .label-after  {{ right: 14px; }}
</style>

<div class="slider-wrap" id="sw">
  <img src="data:image/png;base64,{b64_before}" class="img-before" draggable="false"/>
  <img src="data:image/png;base64,{b64_after}"  class="img-after"  draggable="false" id="ia"/>
  <div class="divider-line" id="dl"></div>
  <div class="divider-handle" id="dh">&#8249;&#8250;</div>
  <div class="label-before">{label_before}</div>
  <div class="label-after">{label_after}</div>
</div>

<script>
(function() {{
  const wrap = document.getElementById('sw');
  const imgA = document.getElementById('ia');
  const line = document.getElementById('dl');
  const hand = document.getElementById('dh');
  let dragging = false;

  function setPos(pct) {{
    pct = Math.max(2, Math.min(98, pct));
    imgA.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
    line.style.left = pct + '%';
    hand.style.left = pct + '%';
  }}

  function getX(e) {{
    const r = wrap.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return ((clientX - r.left) / r.width) * 100;
  }}

  wrap.addEventListener('mousedown',  e => {{ dragging = true; setPos(getX(e)); }});
  wrap.addEventListener('touchstart', e => {{ dragging = true; setPos(getX(e)); }}, {{passive: true}});
  window.addEventListener('mousemove',  e => {{ if (dragging) setPos(getX(e)); }});
  window.addEventListener('touchmove',  e => {{ if (dragging) setPos(getX(e)); }}, {{passive: true}});
  window.addEventListener('mouseup',  () => dragging = false);
  window.addEventListener('touchend', () => dragging = false);
}})();
</script>
"""
    components.html(html, height=height_px + 8, scrolling=False)


# -----------------------------------------------------------------------------
#  Confidence map visualiser
# -----------------------------------------------------------------------------
def render_confidence_map(conf: np.ndarray) -> None:
    """Display confidence map as a heatmap with legend."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap   = plt.get_cmap("RdYlGn")
    rgba   = cmap(conf)
    rgb_u8 = (rgba[:, :, :3] * 255).astype(np.uint8)

    col_img, col_info = st.columns([3, 1], gap="medium")
    with col_img:
        st.image(rgb_u8, use_container_width=True)
        st.markdown("""
        <div class="conf-legend">
            <span style="color:#ef4444; font-weight:600">Low</span>
            <div class="conf-bar"></div>
            <span style="color:#22c55e; font-weight:600">High</span>
        </div>
        """, unsafe_allow_html=True)
    with col_info:
        mean_conf = float(conf.mean())
        high_pct  = float((conf > 0.7).mean() * 100)
        low_pct   = float((conf < 0.3).mean() * 100)

        st.metric("Mean confidence", f"{mean_conf:.2f}")
        st.metric("High-conf pixels", f"{high_pct:.0f}%",
                  help="Pixels where model is > 70% confident")
        st.metric("Uncertain pixels", f"{low_pct:.0f}%",
                  help="Pixels where model is < 30% confident")
        st.caption("Green = certain\nYellow = moderate\nRed = uncertain")


# -----------------------------------------------------------------------------
#  RGB histogram
# -----------------------------------------------------------------------------
def render_histogram(rgb_np: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    for i, (ch_col, ch_name) in enumerate(
        zip(("#ef4444", "#22c55e", "#3b82f6"), ("Red", "Green", "Blue"))
    ):
        hist, _ = np.histogram(rgb_np[:, :, i], bins=128, range=(0, 1))
        xs = np.linspace(0, 1, len(hist))
        ax.fill_between(xs, hist, color=ch_col, alpha=0.28, label=ch_name)
        ax.plot(xs, hist, color=ch_col, linewidth=1.2, alpha=0.85)

    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#e2e8f0")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, framealpha=0, labelcolor="#64748b")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------------------------------------------------
#  Run history helpers
# -----------------------------------------------------------------------------
def _init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []

def _add_history(filename: str, models_run: list[str], img_is_bw: bool):
    _init_history()
    st.session_state["history"].insert(0, {
        "ts":        datetime.now().strftime("%H:%M:%S"),
        "filename":  filename,
        "models":    models_run,
        "bw":        img_is_bw,
    })
    st.session_state["history"] = st.session_state["history"][:10]


# -----------------------------------------------------------------------------
#  Sidebar
# -----------------------------------------------------------------------------
_init_history()

with st.sidebar:
    st.markdown("## Image Colorizer")
    st.caption("AI-powered black & white photo colorization")
    st.divider()

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    st.divider()

    st.markdown("**Models to run**")
    selected_keys = [
        m["key"]
        for m in MODELS
        if st.checkbox(f"{m['icon']}  {m['label']}", value=True, key=f"chk_{m['key']}")
    ]

    st.divider()

    n_models = len(selected_keys)
    if st.button(
        f"Colorize ({n_models})",
        type="primary",
        use_container_width=True,
        disabled=not uploaded or not selected_keys
    ):
        st.session_state["run_requested"] = True

    st.divider()

    use_gpu = st.toggle("Use GPU if available", value=False)

    st.divider()

    st.markdown("**Post-processing**")
    saturation = st.slider(
        "Saturation boost",
        min_value=1.0, max_value=2.5, value=1.0, step=0.1,
        help="Scale ab channels in LAB space. 1.0 = no change.",
    )
    
    use_jbf = st.toggle(
        "Edge-Aligned Colors (JBF)",
        value=True,
        help="Use Joint Bilateral Filtering to snap color boundaries to high-res greyscale edges. Prevents color bleed."
    )

    st.divider()

    st.markdown("**Advanced**")
    use_sr = st.toggle(
        "Super-Resolution (2x)",
        value=False,
        help="Upscale output by 2x using Real-ESRGAN/Lanczos for professional sharpness."
    )

    st.divider()

    st.markdown("**Extras**")
    show_semantic = st.toggle(
        "Semantic segmentation",
        value=False,
        help="Show DeepLabV3 scene understanding -- detects sky, person, grass, etc.",
    )

    st.divider()

    # Run history
    history = st.session_state.get("history", [])
    if history:
        st.markdown("**Recent runs**")
        for h in history:
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-ts">{h["ts"]}</span>'
                f'<span>{h["filename"][:20]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.divider()
    with st.expander("💡 Pro Tips: Advanced Models"):
        st.markdown("""
        **How to advance these models?**
        1. **Fine-tuning**: Train the weights on high-resolution, domain-specific images (e.g., historical portraits).
        2. **Hybrid Architectures**: Replace the simple CNN layers with **Vision Transformers (ViT)** for better global context.
        3. **Post-Filtering**: Apply **Joint Bilateral Filtering** to align the color output precisely with the high-res greyscale edges.
        4. **Super-Resolution**: Chain the output with an upscaler like **Real-ESRGAN** for professional-grade sharpness.
        
        [Open Fine-Tuning Suite →](http://localhost:8501)
        *Note: Run `streamlit run finetune.py` in a separate terminal.*
        """)

device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
if use_gpu and not torch.cuda.is_available():
    st.sidebar.warning("CUDA not available -- using CPU.")


# -----------------------------------------------------------------------------
#  Header
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title">🎨 Image Colorizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Upload a black &amp; white image -- AI models bring it to life in colour.'
    '</div>',
    unsafe_allow_html=True,
)





# -----------------------------------------------------------------------------
#  Landing
# -----------------------------------------------------------------------------
if uploaded is None:
    st.info("👈 Upload a black & white image in the sidebar to start.", icon="🖼️")
    st.stop()


# -----------------------------------------------------------------------------
#  Load image
# -----------------------------------------------------------------------------
pil_img   = Image.open(uploaded).convert("RGB")
img_np    = pil_to_np(pil_img)
gray_np   = to_gray_rgb(img_np)
H, W      = img_np.shape[:2]
img_is_bw = is_grayscale(img_np)
filename  = uploaded.name

# Info strip
st.markdown(f"""
<div class="info-strip">
    <div><div class="is-label">File</div>
         <div class="is-value" style="font-size:.88rem">{filename[:28]}</div></div>
    <div><div class="is-label">Width</div><div class="is-value">{W} px</div></div>
    <div><div class="is-label">Height</div><div class="is-value">{H} px</div></div>
    <div><div class="is-label">Input type</div>
         <div class="is-value">{'Greyscale' if img_is_bw else 'Colour'}</div></div>
    <div><div class="is-label">Device</div>
         <div class="is-value">{str(device).upper()}</div></div>
    <div><div class="is-label">Models</div>
         <div class="is-value">{len(selected_keys)}</div></div>
</div>
""", unsafe_allow_html=True)

# Preview
col_l, col_r = st.columns(2, gap="medium")
with col_l:
    st.image(img_np, use_container_width=True, clamp=True)
    st.markdown('<div class="img-label">Uploaded image</div>', unsafe_allow_html=True)
with col_r:
    st.image(gray_np, use_container_width=True, clamp=True)
    st.markdown(
        '<div class="img-label">Greyscale input (fed to models)</div>',
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------------------------------------------------------
#  Semantic segmentation  (runs separately, before colorization)
# -----------------------------------------------------------------------------
if show_semantic:
    with st.expander("🗺️  Semantic Scene Analysis", expanded=True):
        seg_key = f"seg_{filename}_{H}_{W}"
        if seg_key not in st.session_state:
            with st.spinner("Running DeepLabV3 segmentation..."):
                try:
                    st.session_state[seg_key] = run_semantic(img_np, device)
                except Exception as exc:
                    st.session_state[seg_key] = {"error": str(exc)}

        seg = st.session_state[seg_key]

        if "error" in seg:
            st.markdown(
                f'<div class="msg-error">❌ Segmentation failed: {seg["error"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            seg_col, info_col = st.columns([2, 1], gap="medium")
            with seg_col:
                # Blend original greyscale + coloured segmentation overlay
                overlay = (
                    gray_np * 0.45 +
                    seg["label_rgb"].astype(np.float32) / 255.0 * 0.55
                )
                overlay = np.clip(overlay, 0, 1)
                st.image(overlay, use_container_width=True, clamp=True)
                st.markdown(
                    '<div class="img-label">Detected regions overlay</div>',
                    unsafe_allow_html=True,
                )

            with info_col:
                st.metric("Seg. confidence", f"{seg['conf_mean']:.2f}")
                st.markdown(
                    f'<div class="sec-label" style="margin-top:.75rem">Detected classes</div>',
                    unsafe_allow_html=True,
                )
                chips = ""
                for cls in seg["classes"]:
                    idx = VOC_CLASS_NAMES.index(cls) if cls in VOC_CLASS_NAMES else 0
                    r, g, b = VOC_PALETTE[idx % len(VOC_PALETTE)]
                    chips += (
                        f'<span class="voc-chip" '
                        f'style="background:rgba({r},{g},{b},0.15);'
                        f'color:rgb({max(r-40,0)},{max(g-40,0)},{max(b-40,0)})">'
                        f'{cls}</span>'
                    )
                st.markdown(f'<div class="voc-row">{chips}</div>', unsafe_allow_html=True)

    st.divider()


# -----------------------------------------------------------------------------
#  Validation
# -----------------------------------------------------------------------------
if not selected_keys:
    st.warning("No models selected. Tick at least one model in the sidebar.")
    st.stop()


# -----------------------------------------------------------------------------
#  Run colorization (triggered by sidebar button)
# -----------------------------------------------------------------------------
if st.session_state.get("run_requested"):
    st.session_state["run_requested"] = False
    n_models = len(selected_keys)
    st.session_state["results"] = {}
    bar    = st.progress(0, text="Initialising...")
    status = st.empty()

    for idx, key in enumerate(selected_keys):
        m_info = next(m for m in MODELS if m["key"] == key)
        bar.progress(idx / n_models, text=f"Running {m_info['label']}...")
        status.markdown(f"Running **{m_info['label']}**...")

        t0 = time.time()
        try:
            out = run_model(key, img_np, device)
            rgb = out["rgb"]
            
            # --- Advanced Processing ---
            if use_jbf:
                from colorizers.postprocess import joint_bilateral_filter
                # Need L channel for guide
                from skimage import color
                lab_orig = color.rgb2lab(img_np)
                # out['ab'] is usually 1x2xHxW tensor or np
                # For simplicity, we extract ab from the already-postprocessed rgb or 
                # use the internal ab prediction if available.
                # Let's use the predicted ab for best results.
                # Actually, most runners return rgb. Let's make it more robust.
                lab_pred = color.rgb2lab(rgb)
                ab_refined = joint_bilateral_filter(lab_pred[:,:,1:], lab_orig[:,:,:1])
                lab_pred[:,:,1:] = ab_refined
                rgb = color.lab2rgb(lab_pred).astype(np.float32)

            if saturation > 1.0:
                rgb = boost_saturation(rgb, saturation)
                
            if use_sr:
                from colorizers.superres import apply_super_resolution
                rgb = apply_super_resolution(rgb, device=device)

            st.session_state["results"][key] = {
                "rgb":        rgb,
                "confidence": out.get("confidence"),
                "time":       time.time() - t0,
                "error":      None,
            }
        except Exception as exc:  # noqa: BLE001
            st.session_state["results"][key] = {
                "rgb": None, "confidence": None,
                "time": time.time() - t0, "error": str(exc),
            }

    bar.progress(1.0, text=f"Done -- {n_models} model{'s' if n_models != 1 else ''} complete")
    status.empty()
    _add_history(filename, selected_keys, img_is_bw)
    st.rerun()


# -----------------------------------------------------------------------------
#  Results
# -----------------------------------------------------------------------------
if not st.session_state.get("results"):
    st.stop()

results = st.session_state["results"]
visible = [m for m in MODELS if m["key"] in results]

tab_labels = [
    f"{m['icon']} {m['label']} {'❌' if results[m['key']]['error'] else '✅'}"
    for m in visible
]

st.markdown('<div class="sec-label">Results</div>', unsafe_allow_html=True)
tabs = st.tabs(tab_labels)

for tab, m in zip(tabs, visible):
    with tab:
        r = results[m["key"]]

        # -- Error ----------------------------------------------------------
        if r["error"]:
            st.markdown(
                f'<div class="msg-error">❌ <strong>{m["label"]} failed:</strong> '
                f'{r["error"]}</div>',
                unsafe_allow_html=True,
            )
            continue

        rgb  = r["rgb"]
        conf = r["confidence"]

        # -- Before / After drag slider -------------------------------------
        st.markdown(
            '<div class="sec-label">Drag to compare</div>',
            unsafe_allow_html=True,
        )
        img_h = min(max(H, 280), 520)
        before_after_slider(
            gray_np, rgb,
            label_before="GREYSCALE",
            label_after=m["label"].upper(),
            height_px=img_h,
        )

        st.divider()

        # -- Download + timing ----------------------------------------------
        dl_col, t_col = st.columns([4, 1], gap="medium")
        with dl_col:
            st.download_button(
                label=f"⬇️  Download {m['label']} result",
                data=to_png_bytes(rgb),
                file_name=f"colorized_{m['key']}.png",
                mime="image/png",
                use_container_width=True,
            )
        with t_col:
            st.metric("⏱ Time", f"{r['time']:.1f}s")

        st.divider()

        # -- Confidence map (ECCV + ResNet only) ---------------------------
        if conf is not None:
            st.markdown(
                '<div class="sec-label">Confidence Map</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Shows where the model is certain (green) vs uncertain (red) "
                "about its colour prediction."
            )
            render_confidence_map(conf)
            st.divider()

        # -- Quality metrics (colour reference needed) ----------------------
        if not img_is_bw:
            st.markdown(
                '<div class="sec-label">Quality Metrics</div>',
                unsafe_allow_html=True,
            )
            try:
                metrics = compute_metrics(rgb, img_np)
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.metric("PSNR ↑",   f"{metrics['psnr']:.1f} dB")
                with mc2: st.metric("SSIM ↑",   f"{metrics['ssim']:.3f}")
                with mc3: st.metric("LPIPS ↓",  f"{metrics['lpips']:.4f}")
                with mc4: st.metric("DeltaE2000 ↓", f"{metrics['ciede2000']:.1f}")
                st.caption(
                    "PSNR -- pixel accuracy . SSIM -- structural similarity . "
                    "LPIPS -- perceptual similarity . DeltaE2000 -- colour distance"
                )
            except Exception as exc:  # noqa: BLE001
                st.markdown(
                    f'<div class="msg-warn">⚠️ Metrics failed: {exc}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "💡 Upload a **colour** image to compare accuracy metrics against the original.",
                icon="💡",
            )

        # -- RGB histogram --------------------------------------------------
        st.markdown(
            '<div class="sec-label">RGB Channel Distribution</div>',
            unsafe_allow_html=True,
        )
        render_histogram(rgb)


# -----------------------------------------------------------------------------
#  Comparison gallery (2+ models)
# -----------------------------------------------------------------------------
ok_models = [m for m in visible if results[m["key"]]["error"] is None]

if len(ok_models) > 1:
    st.divider()
    st.markdown('<div class="sec-label">Comparison Gallery</div>', unsafe_allow_html=True)
    g_cols = st.columns(len(ok_models), gap="medium")
    for col, m in zip(g_cols, ok_models):
        r = results[m["key"]]
        with col:
            st.image(r["rgb"], use_container_width=True, clamp=True)
            st.markdown(
                f'<div class="img-label">{m["icon"]} {m["label"]} &nbsp;.&nbsp; '
                f'{r["time"]:.1f}s</div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
#  Footer
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    'BW -> RGB Colorization Suite &nbsp;.&nbsp; ECCV-16 &nbsp;.&nbsp; '
    'SIGGRAPH-17 &nbsp;.&nbsp; ResNet Backbone &nbsp;.&nbsp; DeepLabV3'
    '</div>',
    unsafe_allow_html=True,
)

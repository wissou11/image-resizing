# app.py
import io
import time
import numpy as np
from PIL import Image
import streamlit as st

# =========================
# Core seam-carving code
# =========================

def bytes_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)

def rgb_array_to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()

def to_grayscale(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b

def compute_energy(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return np.abs(dx) + np.abs(dy)

def cumulative_energy_map_fast(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = E.shape
    M = E.astype(np.float32).copy()
    parent = np.full((H, W), -1, dtype=np.int32)

    for y in range(1, H):
        prev = M[y - 1]
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        left[0] = np.inf
        right[-1] = np.inf

        stacked = np.vstack([left, prev, right])
        argmin = np.argmin(stacked, axis=0)
        best = stacked[argmin, np.arange(W)]

        M[y] += best
        parent[y] = np.arange(W) + (argmin - 1)

    return M, parent

def find_vertical_seam_fast(E: np.ndarray) -> np.ndarray:
    M, parent = cumulative_energy_map_fast(E)
    H, _ = E.shape
    seam = np.zeros(H, dtype=np.int32)
    seam[H - 1] = int(np.argmin(M[H - 1]))
    for y in range(H - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]
    return seam

def remove_vertical_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False
    return img[mask].reshape(H, W - 1, C)

def seam_overlay_rgb(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[np.arange(out.shape[0]), seam] = np.array([255, 0, 0], dtype=np.uint8)
    return out

def energy_to_uint8(E: np.ndarray) -> np.ndarray:
    E_norm = E - E.min()
    denom = E_norm.max() + 1e-9
    return (255.0 * (E_norm / denom)).astype(np.uint8)

def seam_carve_width(img: np.ndarray, new_width: int, progress_cb=None) -> np.ndarray:
    H, W, _ = img.shape
    if new_width >= W:
        raise ValueError(f"new_width must be < current width (current W={W}).")

    carved = img.copy()
    seams_to_remove = W - new_width

    for i in range(seams_to_remove):
        gray = to_grayscale(carved)
        E = compute_energy(gray)
        seam = find_vertical_seam_fast(E)
        carved = remove_vertical_seam_fast(carved, seam)

        if progress_cb:
            progress_cb(i + 1, seams_to_remove)

    return carved


# =========================
# UI
# =========================

st.set_page_config(
    page_title="Seam Carving Studio",
    page_icon="🧵",
    layout="wide"
)

# --- little CSS polish (cards, spacing, buttons) ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1 {margin-bottom: 0.2rem;}
.small-note {color: #6b7280; font-size: 0.9rem;}
.metric-card {padding: 0.8rem; border: 1px solid rgba(120,120,120,0.2); border-radius: 16px;}
</style>
""", unsafe_allow_html=True)

st.title("🧵 Seam Carving Studio")
st.markdown('<div class="small-note">Content-aware resizing: remove low-energy seams to preserve important objects.</div>',
            unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    st.divider()
    speed_mode = st.toggle("Speed mode (auto downscale)", value=True)
    max_w = st.slider("Max width (speed mode)", 400, 1600, 1000, 50, disabled=not speed_mode)

    st.divider()
    show_debug = st.toggle("Show debug (energy + first seam)", value=True)
    show_stats = st.toggle("Show stats", value=True)

    st.divider()
    st.caption("Tip: Start with big reduction like 50% for quick demo, then fine-tune.")


if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

# Decode input
orig = bytes_to_rgb_array(uploaded.read())

# optional downscale for speed
if speed_mode and orig.shape[1] > max_w:
    pil = Image.fromarray(orig)
    new_h = int(orig.shape[0] * (max_w / orig.shape[1]))
    pil = pil.resize((max_w, new_h), Image.Resampling.LANCZOS)
    orig = np.array(pil, dtype=np.uint8)
    st.warning(f"Speed mode: image downscaled to {orig.shape[1]}×{orig.shape[0]} for faster processing.")

H, W, _ = orig.shape

# Top row: controls + preview
topL, topR = st.columns([1.2, 2.2], vertical_alignment="top")

with topL:
    st.subheader("🎯 Target width")
    target_mode = st.radio("Choose mode", ["Percent", "Pixels"], horizontal=True)

    if target_mode == "Percent":
        pct = st.slider("Keep % of original width", 50, 99, 50, 1)
        target_width = int(W * (pct / 100.0))
    else:
        target_width = st.slider("Width (px)", min_value=max(1, W//2), max_value=W-1, value=max(1, W//2), step=1)

    seams_to_remove = W - target_width
    st.markdown(f"**Remove:** `{seams_to_remove}` seams")

    run = st.button("🚀 Run Seam Carving", type="primary", use_container_width=True)

    if show_stats:
        st.markdown("### 📊 Stats")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-card">Original<br><b>{}×{}</b></div>'.format(W, H), unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">Target<br><b>{}×{}</b></div>'.format(target_width, H), unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">Seams<br><b>{}</b></div>'.format(seams_to_remove), unsafe_allow_html=True)

with topR:
    st.subheader("🖼️ Preview")
    st.image(orig, caption=f"Original ({W}×{H})", use_container_width=True)

if not run:
    st.stop()

# Debug visuals (first seam only)
debug_energy_img = None
debug_seam_img = None
if show_debug:
    gray0 = to_grayscale(orig)
    E0 = compute_energy(gray0)
    seam0 = find_vertical_seam_fast(E0)
    debug_energy_img = energy_to_uint8(E0)
    debug_seam_img = seam_overlay_rgb(orig, seam0)

# Progress + run
progress = st.progress(0)
status = st.empty()

t0 = time.time()

def progress_cb(done, total):
    if total <= 0:
        progress.progress(100)
        return
    p = int((done / total) * 100)
    progress.progress(p)
    status.write(f"Removing seams: {done}/{total} ({p}%)")

try:
    carved = seam_carve_width(orig, target_width, progress_cb=progress_cb)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

dt = time.time() - t0
status.success(f"Done ✅ ({dt:.2f}s)")

# Results row
st.divider()
st.subheader("✅ Before / After")

c1, c2 = st.columns(2)
with c1:
    st.image(orig, caption=f"Before ({W}×{H})", use_container_width=True)
with c2:
    st.image(carved, caption=f"After ({carved.shape[1]}×{carved.shape[0]})", use_container_width=True)

# Debug in expander (clean)
if show_debug and debug_energy_img is not None:
    with st.expander("🔍 Debug visuals (first seam)", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.image(debug_energy_img, caption="Energy map (normalized)", use_container_width=True)
        with d2:
            st.image(debug_seam_img, caption="First seam overlay (red)", use_container_width=True)

# Download
st.divider()
st.subheader("⬇️ Download")
png_bytes = rgb_array_to_png_bytes(carved)
st.download_button(
    "Download result as PNG",
    data=png_bytes,
    file_name="seam_carved.png",
    mime="image/png",
    use_container_width=True
)

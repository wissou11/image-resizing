# app.py
import io
import numpy as np
from PIL import Image
import streamlit as st

# ============================================================
# 1) I/O helpers (Streamlit versions)
# ============================================================

def bytes_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to RGB uint8 array (H, W, 3)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)

def rgb_array_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode RGB uint8 array (H, W, 3) to PNG bytes."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32 (H, W)."""
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b

# ============================================================
# 2) ENERGY MAP
# ============================================================

def compute_energy(gray: np.ndarray) -> np.ndarray:
    """
    Compute energy as |dx| + |dy| using central differences.
    Border handling: edge padding.
    """
    padded = np.pad(gray, 1, mode="edge")
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return np.abs(dx) + np.abs(dy)

def energy_to_uint8(E: np.ndarray) -> np.ndarray:
    """Normalize energy map to uint8 0..255 for display."""
    E_norm = E - E.min()
    denom = (E_norm.max() + 1e-9)
    E_norm = (255.0 * (E_norm / denom)).astype(np.uint8)
    return E_norm

# ============================================================
# 3) FAST SHORTEST PATH ON DAG (DP vectorized per row)
# ============================================================

def cumulative_energy_map_fast(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cumulative energy M and parent pointers (for seam backtracking).
    """
    H, W = E.shape
    M = E.astype(np.float32).copy()
    parent = np.full((H, W), -1, dtype=np.int32)

    for y in range(1, H):
        prev = M[y - 1]

        left = np.roll(prev, 1)
        right = np.roll(prev, -1)

        left[0] = np.inf
        right[-1] = np.inf

        stacked = np.vstack([left, prev, right])  # (3, W)
        argmin = np.argmin(stacked, axis=0)       # 0=left, 1=up, 2=right
        best = stacked[argmin, np.arange(W)]

        M[y] += best
        parent[y] = np.arange(W) + (argmin - 1)   # -1,0,+1

    return M, parent

def find_vertical_seam_fast(E: np.ndarray) -> np.ndarray:
    """Return seam indices (H,), seam[y] = x to remove."""
    M, parent = cumulative_energy_map_fast(E)
    H, _ = E.shape

    seam = np.zeros(H, dtype=np.int32)
    seam[H - 1] = int(np.argmin(M[H - 1]))

    for y in range(H - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]

    return seam

# ============================================================
# 4) FAST SEAM REMOVAL (boolean mask)
# ============================================================

def remove_vertical_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove seam from RGB image (H,W,3) -> (H,W-1,3)."""
    H, W, C = img.shape
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False
    return img[mask].reshape(H, W - 1, C)

# ============================================================
# 5) VISUALS (energy map + seam overlay) for Streamlit
# ============================================================

def seam_overlay_rgb(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Return a copy with seam highlighted in red."""
    out = img.copy()
    out[np.arange(out.shape[0]), seam] = np.array([255, 0, 0], dtype=np.uint8)
    return out

# ============================================================
# 6) MAIN RESIZING LOOP
# ============================================================

def seam_carve_width(img: np.ndarray, new_width: int) -> np.ndarray:
    """Reduce width to new_width using seam carving (reduction only)."""
    H, W, _ = img.shape
    if new_width >= W:
        raise ValueError(f"new_width must be < current width (current W={W}).")

    carved = img.copy()
    seams_to_remove = W - new_width

    for _ in range(seams_to_remove):
        gray = to_grayscale(carved)
        E = compute_energy(gray)
        seam = find_vertical_seam_fast(E)
        carved = remove_vertical_seam_fast(carved, seam)

    return carved

# ============================================================
# 7) STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Seam Carving", layout="wide")
st.title("Seam Carving — Content-Aware Image Resizing")

with st.sidebar:
    st.header("Settings")
    show_debug = st.checkbox("Show energy map + first seam overlay", value=True)
    max_width_limit = st.checkbox("Limit max input width (faster)", value=True)
    max_w = st.slider("Max allowed input width", 300, 1400, 900, 50, disabled=not max_width_limit)

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

# Read & decode
orig = bytes_to_rgb_array(uploaded.read())

# Optional downscale for speed (keeps aspect ratio)
if max_width_limit and orig.shape[1] > max_w:
    pil = Image.fromarray(orig)
    new_h = int(orig.shape[0] * (max_w / orig.shape[1]))
    pil = pil.resize((max_w, new_h), Image.Resampling.LANCZOS)
    orig = np.array(pil, dtype=np.uint8)
    st.warning(f"Image downscaled to width={max_w} for speed.")

H, W, _ = orig.shape

st.subheader("Choose target width")
colA, colB = st.columns([2, 1])

with colB:
    target = st.slider(
        "Target width (pixels)",
        min_value=max(1, W // 2),  # half minimum to avoid crazy shrink
        max_value=W - 1,
        value=max(1, W // 2),
        step=1
    )
    run = st.button("Run Seam Carving", type="primary")

with colA:
    st.image(orig, caption=f"Original ({W}×{H})", use_container_width=True)

if not run:
    st.stop()

# Debug visuals for FIRST seam (computed once)
debug_energy_img = None
debug_seam_img = None

if show_debug:
    gray0 = to_grayscale(orig)
    E0 = compute_energy(gray0)
    seam0 = find_vertical_seam_fast(E0)
    debug_energy_img = energy_to_uint8(E0)              # (H,W) uint8
    debug_seam_img = seam_overlay_rgb(orig, seam0)      # (H,W,3)

# Perform carving with progress
seams_to_remove = W - target
carved = orig.copy()

progress = st.progress(0)
status = st.empty()

for i in range(seams_to_remove):
    gray = to_grayscale(carved)
    E = compute_energy(gray)
    seam = find_vertical_seam_fast(E)
    carved = remove_vertical_seam_fast(carved, seam)

    if seams_to_remove > 0:
        p = int(((i + 1) / seams_to_remove) * 100)
        progress.progress(p)
        status.write(f"Removing seams: {i+1}/{seams_to_remove}")

status.write("Done ✅")

# Show results side-by-side
st.subheader("Result")
c1, c2 = st.columns(2)
with c1:
    st.image(orig, caption=f"Before ({W}×{H})", use_container_width=True)
with c2:
    st.image(carved, caption=f"After ({carved.shape[1]}×{carved.shape[0]})", use_container_width=True)

# Debug section
if show_debug and debug_energy_img is not None:
    st.subheader("Debug (first seam)")
    d1, d2 = st.columns(2)
    with d1:
        st.image(debug_energy_img, caption="Energy map (normalized)", use_container_width=True)
    with d2:
        st.image(debug_seam_img, caption="First seam overlay (red)", use_container_width=True)

# Download
st.subheader("Download")
png_bytes = rgb_array_to_png_bytes(carved)
st.download_button(
    "Download carved image (PNG)",
    data=png_bytes,
    file_name="seam_carved.png",
    mime="image/png"
)

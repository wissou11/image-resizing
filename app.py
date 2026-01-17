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
# Streamlit UI + Advanced CSS
# =========================

st.set_page_config(page_title="Seam Carving Studio", page_icon="🧵", layout="wide")

st.markdown("""
<style>
/* ---------- Global layout ---------- */
:root{
  --bg1: #0b1020;
  --bg2: #0b1224;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.12);
  --stroke2: rgba(255,255,255,0.18);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.62);
  --muted2: rgba(255,255,255,0.48);
  --accent: #7c3aed;        /* violet */
  --accent2: #22c55e;       /* green */
  --warn: #f59e0b;          /* amber */
  --danger: #ef4444;        /* red */
  --shadow: 0 18px 50px rgba(0,0,0,0.45);
  --shadow2: 0 12px 30px rgba(0,0,0,0.35);
  --radius: 20px;
  --radius2: 16px;
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 20% -10%, rgba(124,58,237,0.45), transparent 60%),
    radial-gradient(1000px 600px at 110% 10%, rgba(34,197,94,0.35), transparent 55%),
    radial-gradient(900px 700px at 45% 120%, rgba(59,130,246,0.25), transparent 55%),
    linear-gradient(180deg, var(--bg1), var(--bg2)) !important;
  color: var(--text) !important;
}

/* reduce default top padding */
.block-container{
  padding-top: 1.2rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1250px;
}

/* ---------- Header ---------- */
.app-hero{
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 22px 22px;
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  margin-bottom: 18px;
}
.app-hero:before{
  content:"";
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(500px 200px at 20% 0%, rgba(124,58,237,0.30), transparent 60%),
    radial-gradient(500px 250px at 85% 20%, rgba(34,197,94,0.22), transparent 60%);
  filter: blur(10px);
  opacity: 0.9;
  z-index:0;
}
.app-hero *{ position: relative; z-index:1; }
.hero-title{
  font-size: 38px;
  font-weight: 800;
  letter-spacing: -0.03em;
  margin: 0;
  line-height: 1.05;
}
.hero-sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 14.5px;
  line-height: 1.55;
}
.badges{
  margin-top: 14px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}
.badge{
  border: 1px solid var(--stroke2);
  background: rgba(255,255,255,0.06);
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 12.5px;
  color: var(--muted);
  backdrop-filter: blur(10px);
}

/* ---------- Cards ---------- */
.card{
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.05);
  border-radius: var(--radius);
  padding: 16px 16px;
  box-shadow: var(--shadow2);
}
.card-title{
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
  color: var(--text);
}
.card-note{
  margin-top: 8px;
  font-size: 12.5px;
  color: var(--muted2);
  line-height: 1.45;
}
.metric-row{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top: 10px;
}
.metric{
  flex: 1 1 160px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.045);
  border-radius: 16px;
  padding: 12px 12px;
}
.metric .k{
  font-size: 12px;
  color: var(--muted2);
}
.metric .v{
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.02em;
}

/* ---------- Streamlit elements ---------- */
[data-testid="stFileUploader"]{
  border: 1px dashed rgba(255,255,255,0.25);
  border-radius: var(--radius);
  padding: 10px 10px;
  background: rgba(255,255,255,0.04);
}
[data-testid="stFileUploader"] section{
  background: transparent !important;
}
[data-testid="stSlider"]{
  padding: 6px 8px;
  border-radius: 14px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
}
.stButton>button{
  width: 100%;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.18);
  padding: 0.75rem 1rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(34,197,94,0.75)) !important;
  box-shadow: 0 14px 35px rgba(0,0,0,0.35);
  transition: transform .08s ease, filter .08s ease;
}
.stButton>button:hover{
  transform: translateY(-1px);
  filter: brightness(1.05);
}
.stButton>button:active{
  transform: translateY(0px) scale(0.99);
}
div[data-testid="stDownloadButton"] button{
  background: linear-gradient(135deg, rgba(59,130,246,0.85), rgba(124,58,237,0.85)) !important;
}

/* progress bar */
[data-testid="stProgress"] > div > div{
  border-radius: 999px;
}

/* images look like cards */
[data-testid="stImage"] img{
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 18px 45px rgba(0,0,0,0.35);
}

/* expander */
details{
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 8px 10px;
}

/* hide streamlit default menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-hero">
  <div class="hero-title">🧵 Seam Carving Studio</div>
  <div class="hero-sub">
    Redimensionnement <b>content-aware</b> : on enlève des “seams” de faible énergie pour préserver les zones importantes.
    Parfait pour une démo + rapport (energy map + seam overlay).
  </div>
  <div class="badges">
    <div class="badge">⚡ DP vectorisé</div>
    <div class="badge">🖼️ Upload → Resize → Download</div>
    <div class="badge">🔍 Debug (1er seam)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main area uploader (NOT sidebar)
uploader_col, settings_col = st.columns([1.6, 1.0], vertical_alignment="top")

with uploader_col:
    st.markdown('<div class="card"><div class="card-title">1) Upload image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Browse a JPG/PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('<div class="card-note">Astuce: si le traitement est lent, active “Speed mode” et limite la largeur.</div></div>', unsafe_allow_html=True)

with settings_col:
    st.markdown('<div class="card"><div class="card-title">2) Options</div>', unsafe_allow_html=True)
    speed_mode = st.toggle("Speed mode (auto downscale)", value=True)
    max_w = st.slider("Max width (speed mode)", 400, 1800, 1100, 50, disabled=not speed_mode)
    show_debug = st.toggle("Show debug (energy + first seam)", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not uploaded:
    st.stop()

orig = bytes_to_rgb_array(uploaded.read())

# optional downscale for speed
if speed_mode and orig.shape[1] > max_w:
    pil = Image.fromarray(orig)
    new_h = int(orig.shape[0] * (max_w / orig.shape[1]))
    pil = pil.resize((max_w, new_h), Image.Resampling.LANCZOS)
    orig = np.array(pil, dtype=np.uint8)
    st.warning(f"Speed mode: image downscaled to {orig.shape[1]}×{orig.shape[0]} for faster processing.")

H, W, _ = orig.shape

st.markdown("")

controls, preview = st.columns([1.0, 1.8], vertical_alignment="top")

with controls:
    st.markdown('<div class="card"><div class="card-title">3) Target width</div>', unsafe_allow_html=True)

    mode = st.radio("Mode", ["Percent", "Pixels"], horizontal=True)
    if mode == "Percent":
        pct = st.slider("Keep % of width", 50, 99, 50, 1)
        target_width = int(W * (pct / 100.0))
    else:
        target_width = st.slider("Width (px)", min_value=max(1, W//2), max_value=W-1, value=max(1, W//2), step=1)

    seams_to_remove = W - target_width

    st.markdown(f"<div class='card-note'>Remove <b>{seams_to_remove}</b> seams</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    st.markdown(f"""
      <div class="metric"><div class="k">Original</div><div class="v">{W}×{H}</div></div>
      <div class="metric"><div class="k">Target</div><div class="v">{target_width}×{H}</div></div>
      <div class="metric"><div class="k">Seams</div><div class="v">{seams_to_remove}</div></div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("🚀 Run Seam Carving", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

with preview:
    st.markdown('<div class="card"><div class="card-title">Preview</div>', unsafe_allow_html=True)
    st.image(orig, caption=f"Original ({W}×{H})", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not run:
    st.stop()

# Debug first seam
debug_energy_img = None
debug_seam_img = None
if show_debug:
    gray0 = to_grayscale(orig)
    E0 = compute_energy(gray0)
    seam0 = find_vertical_seam_fast(E0)
    debug_energy_img = energy_to_uint8(E0)
    debug_seam_img = seam_overlay_rgb(orig, seam0)

# Run with progress
progress = st.progress(0)
status = st.empty()

t0 = time.time()

def progress_cb(done, total):
    p = 100 if total <= 0 else int((done / total) * 100)
    progress.progress(p)
    status.write(f"Processing: {done}/{total} ({p}%)")

try:
    carved = seam_carve_width(orig, target_width, progress_cb=progress_cb)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

dt = time.time() - t0
status.success(f"Done ✅ ({dt:.2f}s)")

st.markdown("")

st.markdown('<div class="card"><div class="card-title">✅ Before / After</div>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    st.image(orig, caption=f"Before ({W}×{H})", use_container_width=True)
with b2:
    st.image(carved, caption=f"After ({carved.shape[1]}×{carved.shape[0]})", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if show_debug and debug_energy_img is not None:
    with st.expander("🔍 Debug visuals (first seam)"):
        d1, d2 = st.columns(2)
        with d1:
            st.image(debug_energy_img, caption="Energy map (normalized)", use_container_width=True)
        with d2:
            st.image(debug_seam_img, caption="First seam overlay (red)", use_container_width=True)

st.markdown("")

st.markdown('<div class="card"><div class="card-title">⬇️ Download</div>', unsafe_allow_html=True)
png_bytes = rgb_array_to_png_bytes(carved)
st.download_button(
    "Download result as PNG",
    data=png_bytes,
    file_name="seam_carved.png",
    mime="image/png",
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

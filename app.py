import numpy as np
from PIL import Image


# ============================================================
# 1) I/O
# ============================================================

def load_image(path: str) -> np.ndarray:
    """Load an image as uint8 array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(img: np.ndarray, path: str) -> None:
    """Save an image array (H, W, 3)."""
    Image.fromarray(img).save(path)


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


# ============================================================
# 3) FAST SHORTEST PATH ON DAG (DP vectorized per row)
# ============================================================

def cumulative_energy_map_fast(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cumulative energy M and parent pointers (for seam backtracking).

    Vectorized per row:
      M[y, x] = E[y, x] + min(M[y-1, x-1], M[y-1, x], M[y-1, x+1])
      parent[y, x] stores arg parent column in row y-1.
    """
    H, W = E.shape
    M = E.astype(np.float32).copy()
    parent = np.full((H, W), -1, dtype=np.int32)

    for y in range(1, H):
        prev = M[y - 1]

        left = np.roll(prev, 1)
        right = np.roll(prev, -1)

        # Invalidate wrap-around neighbors at borders
        left[0] = np.inf
        right[-1] = np.inf

        stacked = np.vstack([left, prev, right])  # shape (3, W)
        argmin = np.argmin(stacked, axis=0)       # 0=left, 1=up, 2=right
        best = stacked[argmin, np.arange(W)]

        M[y] += best
        parent[y] = np.arange(W) + (argmin - 1)   # shift: -1, 0, +1

    return M, parent


def find_vertical_seam_fast(E: np.ndarray) -> np.ndarray:
    """
    Find minimum-energy vertical seam.
    Returns seam: int32 array (H,), seam[y] = x index to remove in row y.
    """
    M, parent = cumulative_energy_map_fast(E)
    H, W = E.shape

    seam = np.zeros(H, dtype=np.int32)
    seam[H - 1] = int(np.argmin(M[H - 1]))

    for y in range(H - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]

    return seam


# ============================================================
# 4) FAST SEAM REMOVAL (boolean mask)
# ============================================================

def remove_vertical_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Remove a vertical seam from an RGB image using a boolean mask.
    img: (H, W, 3), seam: (H,)
    Returns: (H, W-1, 3)
    """
    H, W, C = img.shape
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False
    return img[mask].reshape(H, W - 1, C)


# ============================================================
# 5) OPTIONAL: DEBUG VISUALS (energy map + seam overlay)
# ============================================================

def save_energy_map_image(E: np.ndarray, path: str) -> None:
    """
    Save energy map as a grayscale image (normalized to 0..255).
    Useful for report figures.
    """
    E_norm = E - E.min()
    denom = (E_norm.max() + 1e-9)
    E_norm = (255.0 * (E_norm / denom)).astype(np.uint8)
    Image.fromarray(E_norm, mode="L").save(path)


def save_seam_overlay(img: np.ndarray, seam: np.ndarray, path: str) -> None:
    """
    Save the image with the seam highlighted in red.
    """
    out = img.copy()
    out[np.arange(out.shape[0]), seam] = np.array([255, 0, 0], dtype=np.uint8)
    save_image(out, path)


# ============================================================
# 6) MAIN RESIZING LOOP
# ============================================================

def seam_carve_width(img: np.ndarray, new_width: int, debug: bool = False) -> np.ndarray:
    """
    Reduce width to new_width using seam carving.
    Only supports reduction (new_width < current width).
    If debug=True, saves energy map and seam overlay for the first seam.
    """
    H, W, _ = img.shape
    if new_width >= W:
        raise ValueError(f"new_width must be < current width (current W={W}).")

    carved = img.copy()
    seams_to_remove = W - new_width

    for i in range(seams_to_remove):
        gray = to_grayscale(carved)
        E = compute_energy(gray)
        seam = find_vertical_seam_fast(E)

        # Save debug visuals only for the first iteration (or adjust as you want)
        if debug and i == 0:
            save_energy_map_image(E, "energy_map1.png")
            save_seam_overlay(carved, seam, "seam1.png")

        carved = remove_vertical_seam_fast(carved, seam)

    return carved


# ============================================================
# 7) RUN EXAMPLE
# ============================================================

if __name__ == "__main__":
    input_path = "giraffe.jpg"
    output_path = "output_carved1.jpg"

    img = load_image(input_path)
    
    target_width =img.shape[1] // 2
    # debug=True will generate:
    # - energy_map.png
    # - seam.png (seam in red)
    out = seam_carve_width(img, target_width, debug=True)

    save_image(img, "original1.jpg")     # for report
    save_image(out, output_path)
    save_image(out, "after1.jpg")        # for report

    print("Done.")
    print("Saved:", output_path, "energy_map.png seam.png original.jpg after.jpg")

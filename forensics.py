"""
forensics.py — Forensic feature extraction + AI-generated heuristic scoring
# NEW
Extracts ELA, FFT, noise, metadata signals and produces visual heatmaps.
"""

import io
import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import fft2, fftshift
from typing import Tuple

from config import ELA_QUALITY, AI_GEN_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# ELA — Error Level Analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_ela(image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute ELA: re-compress at quality=ELA_QUALITY and measure pixel diff.

    Returns:
        ela_avg_diff (float): mean absolute pixel diff → higher = more edited
        ela_map (np.ndarray): per-pixel diff map (uint8, amplified x10)
    """
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=ELA_QUALITY)
    buf.seek(0)

    comp     = Image.open(buf).convert('RGB')
    orig     = pil_img.convert('RGB')
    arr_orig = np.array(orig,  dtype=np.int16)
    arr_comp = np.array(comp,  dtype=np.int16)
    diff_map = np.abs(arr_orig - arr_comp)

    ela_avg_diff = float(diff_map.mean())
    # Amplified map for visualisation
    ela_vis = np.clip(diff_map.mean(axis=2) * 10, 0, 255).astype(np.uint8)

    return ela_avg_diff, ela_vis


# ─────────────────────────────────────────────────────────────────────────────
# Noise Analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_noise(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimate noise via Laplacian residual.

    Returns:
        noise_sigma (float): std of noise residual
        noise_map (np.ndarray): amplified noise map (uint8)
    """
    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    residual  = gray.astype(np.int16) - blur.astype(np.int16)
    noise_sigma = float(np.std(residual))
    noise_map = np.clip(np.abs(residual) * 4, 0, 255).astype(np.uint8)
    return noise_sigma, noise_map


# ─────────────────────────────────────────────────────────────────────────────
# FFT — Frequency Domain Analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_fft(gray: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute FFT magnitude spectrum and return peak/mean metrics + heatmap.

    Returns:
        fft_peak (float): max magnitude (excluding DC component)
        fft_mean (float): mean magnitude (excluding DC)
        fft_heatmap (np.ndarray): uint8 heatmap image (H, W)
    """
    f   = fft2(gray.astype(np.float32))
    mag = 20 * np.log(np.abs(fftshift(f)) + 1)

    r, c = mag.shape
    m   = 30  # mask DC component
    mag_masked = mag.copy()
    mag_masked[r//2 - m: r//2 + m, c//2 - m: c//2 + m] = 0

    fft_peak = float(np.max(mag_masked))
    fft_mean = float(np.mean(mag_masked))

    # Normalise to uint8 for visualisation
    fft_norm = ((mag_masked - mag_masked.min()) /
                (mag_masked.max() - mag_masked.min() + 1e-8) * 255).astype(np.uint8)

    return fft_peak, fft_mean, fft_norm


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_metadata(image_path: str) -> Tuple[bool, bool]:
    """
    Check EXIF metadata for editing software signatures and assess authenticity.

    Returns:
        editing_flag (bool): True if editing software detected in EXIF
        has_exif     (bool): True if any EXIF data present
    """
    from PIL import ExifTags
    editing_keywords = ["Adobe", "Photoshop", "GIMP", "Editor", "Picasa",
                        "Lightroom", "Snapseed", "stable diffusion", "midjourney",
                        "DALL-E", "firefly"]
    try:
        img  = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                tag_name = ExifTags.TAGS.get(tag, "")
                if tag_name == "Software":
                    val_str = str(val).lower()
                    if any(k.lower() in val_str for k in editing_keywords):
                        return True, True
            return False, True   # has EXIF but no editing flag
        return False, False      # no EXIF at all
    except Exception:
        return False, False


# ─────────────────────────────────────────────────────────────────────────────
# AI-Generated Heuristic Scorer
# ─────────────────────────────────────────────────────────────────────────────
def compute_ai_gen_score(
    ela_avg: float,
    fft_peak: float,
    fft_mean: float,
    noise_sigma: float,
    has_exif: bool,
) -> float:
    """
    Compute a heuristic AI-generated image score [0, 1].

    Uses RELATIVE deviation scoring instead of absolute thresholds.
    Baselines are calibrated for typical JPEG-compressed/preprocessed images:
      - FFT ratio baseline ~1.5-2.5 (JPEG compressed images have low ratios)
      - ELA baseline ~1.5-3.0 (JPEG artefacts are present in all JPEG images)
      - Noise baseline ~5-15 (denoised / compressed images have low noise)

    A score is HIGH (towards 1.0) only for images that are EXTREMELY outlier:
      - Near-zero noise (< 3)        → very synthetic
      - Near-zero ELA (< 0.5)        → unrealistically clean
      - FFT ratio < 1.0              → perfectly uniform spectrum
      - Zero EXIF in a JPEG          → likely generated

    Returns:
        ai_gen_score (float): 0 = not AI-generated, 1 = likely AI-generated
    """
    scores = {}

    fft_ratio = fft_peak / (fft_mean + 1e-6)

    # FFT: Real JPEGs typically have ratio 1.5 - 2.5. 
    # AI-gen often have smoother high frequencies (ratio < 2.0).
    fft_score = float(np.clip((2.0 - fft_ratio) / 1.0, 0.0, 1.0))
    scores["fft_uniformity"] = fft_score

    # ELA: Real JPEGs ELA typically > 1.2
    # Pristine AI-gen often < 1.0, but we allow up to 1.8 for compressed AI
    ela_score = float(np.clip((1.8 - ela_avg) / 1.0, 0.0, 1.0))
    scores["ela_uniformity"] = ela_score

    # Noise: Real JPEGs typically > 3.0
    # Pure AI-gen typically < 3.0, allow up to 5.0
    noise_score = float(np.clip((5.0 - noise_sigma) / 3.0, 0.0, 1.0))
    scores["noise_smoothness"] = noise_score

    # EXIF: missing metadata is a mild signal (e.g., screenshot or generated)
    scores["no_metadata"] = 0.0 if has_exif else 0.5

    # Weighted sum
    ai_gen_weights_local = {
        "fft_uniformity":   0.35, # increased back relative to ELA
        "ela_uniformity":   0.35,
        "noise_smoothness": 0.20,
        "no_metadata":      0.10,
    }
    ai_gen_score = sum(
        ai_gen_weights_local[k] * scores[k] for k in ai_gen_weights_local
    )
    return float(np.clip(ai_gen_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Colourmap helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_colormap(gray_map: np.ndarray, colormap=cv2.COLORMAP_JET) -> Image.Image:
    """Convert a uint8 grayscale array to a colourised PIL Image."""
    colored = cv2.applyColorMap(gray_map, colormap)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def run_full_forensics(image_bgr: np.ndarray, gray: np.ndarray, image_path: str) -> dict:
    """
    Convenience wrapper: run all forensic extractors and return a results dict.
    """
    ela_avg, ela_map             = extract_ela(image_bgr)
    noise_sigma, noise_map       = extract_noise(gray)
    fft_peak, fft_mean, fft_map  = extract_fft(gray)
    editing_flag, has_exif       = extract_metadata(image_path)
    ai_gen_score                 = compute_ai_gen_score(ela_avg, fft_peak, fft_mean,
                                                         noise_sigma, has_exif)

    return {
        "ela_avg":      ela_avg,
        "ela_map":      ela_map,
        "noise_sigma":  noise_sigma,
        "noise_map":    noise_map,
        "fft_peak":     fft_peak,
        "fft_mean":     fft_mean,
        "fft_map":      fft_map,
        "editing_flag": editing_flag,
        "has_exif":     has_exif,
        "ai_gen_score": ai_gen_score,
    }

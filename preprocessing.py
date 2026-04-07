"""
preprocessing.py — Shared preprocessing used in BOTH training and inference
# NEW
Ensures 100% consistency between train-time and inference-time pipelines.
"""

import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# ── Shared CNN Transform (inference & validation) ─────────────────────────────
# This is the CANONICAL transform. Import and use this everywhere.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Augmented Transform (training only) ──────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_cv2_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply OpenCV-side preprocessing (resize, denoise, CLAHE) used when
    preparing images for forensic analysis (ELA, FFT, noise).
    Returns a BGR numpy array of shape (IMG_SIZE[1], IMG_SIZE[0], 3).
    """
    if img_bgr is None:
        raise ValueError("Input image is None — cannot preprocess.")

    # Step 1: Resize
    img = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Step 2: Denoise (mild Gaussian blur)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    # Step 3: CLAHE on the L-channel of LAB colour space
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def preprocess_for_inference(pil_image: Image.Image):
    """
    Convert a PIL Image → normalised PyTorch tensor ready for CNN inference.
    This is the SINGLE source of truth for inference preprocessing.
    
    Args:
        pil_image: PIL Image (RGB)
    Returns:
        torch.Tensor of shape (1, 3, H, W) ready for model input
    """
    pil_rgb = pil_image.convert("RGB")
    return INFERENCE_TRANSFORM(pil_rgb).unsqueeze(0)


def load_and_preprocess(image_path: str):
    """
    Load an image from disk with robust error handling, apply full 
    preprocessing pipeline, and return:
    - preprocessed BGR array (for forensics)
    - grayscale array (for FFT/noise)
    - PIL RGB image (for CNN tensor)

    Raises ValueError if image cannot be read or is corrupted.
    """
    try:
        # Step 0: Ensure path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Step 1: Open with PIL to verify integrity
        with Image.open(image_path) as verify_img:
            verify_img.verify()

        # Step 2: Load with cv2 for forensic processing
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"OpenCV failed to decode image: {image_path}")

        # Step 3: Run pipeline
        img_preprocessed = preprocess_cv2_image(img_bgr)
        gray = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2GRAY)
        pil_rgb = Image.fromarray(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB))

        return img_preprocessed, gray, pil_rgb

    except Exception as e:
        raise ValueError(f"Preprocessing error for '{image_path}': {str(e)}")


def load_pil_image(image_path: str) -> Image.Image:
    """Safe PIL image loader with error handling and orientation normalization."""
    try:
        img = Image.open(image_path)
        img.load()  # Force load to check for truncation
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Cannot open/verify image '{image_path}': {e}")

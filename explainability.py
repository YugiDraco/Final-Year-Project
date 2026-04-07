"""
explainability.py — Grad-CAM, FFT heatmap, and noise map visualisations
# NEW
Returns PIL Images for display in the Gradio UI.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple

from config import DEVICE, IMG_SIZE, GRADCAM_ALPHA
from preprocessing import preprocess_for_inference
from forensics import apply_colormap


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    Hooks into the last convolutional block (features[-1]) to get
    activation maps and gradients.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNet-B0: last feature block before AdaptiveAvgPool
        target_layer = self.model.features[-1]   # Block 8 output

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """
        Generate a Grad-CAM heatmap for the input PIL image.

        Returns:
            cam_normalised (np.ndarray): float32 array [0,1] shaped (H, W)
            or None if generation fails.
        """
        tensor = preprocess_for_inference(pil_image).to(DEVICE)
        tensor.requires_grad_(True)

        self.model.eval()
        try:
            logit = self.model(tensor)
            self.model.zero_grad()
            logit.backward(torch.ones_like(logit))
        except Exception as e:
            print(f"[GradCAM] Backward pass failed: {e}")
            return None

        if self.gradients is None or self.activations is None:
            return None

        # Pool gradients across channels
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])   # (C,)
        activations  = self.activations[0]                   # (C, H, W)

        # Weight activations by pooled gradients
        for c in range(activations.shape[0]):
            activations[c, :, :] *= pooled_grads[c]

        cam = activations.cpu().numpy().mean(axis=0)         # (H, W)
        cam = np.maximum(cam, 0)                             # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)


def generate_gradcam_overlay(
    model: nn.Module,
    pil_image: Image.Image,
    alpha: float = GRADCAM_ALPHA,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap and overlay it on the original image.

    Args:
        model:     Loaded EfficientNet-B0 model
        pil_image: Input PIL image (RGB)
        alpha:     Overlay transparency

    Returns:
        PIL Image with Grad-CAM overlay
    """
    gcam = GradCAM(model)
    try:
        cam = gcam.generate(pil_image)
    finally:
        gcam.remove_hooks()

    if cam is None:
        # Return original image if GradCAM fails
        return pil_image.resize(IMG_SIZE)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, IMG_SIZE)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)

    # Apply JET colormap
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original image
    orig_arr = np.array(pil_image.resize(IMG_SIZE).convert("RGB"))

    # Overlay
    overlay  = (alpha * heatmap + (1 - alpha) * orig_arr).astype(np.uint8)
    return Image.fromarray(overlay)


# ─────────────────────────────────────────────────────────────────────────────
# FFT Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def generate_fft_heatmap(gray: np.ndarray) -> Image.Image:
    """
    Compute and return a colour-coded FFT magnitude heatmap.

    Args:
        gray: uint8 grayscale numpy array

    Returns:
        PIL Image (RGB) showing frequency domain magnitude
    """
    from scipy.fftpack import fft2, fftshift
    f   = fft2(gray.astype(np.float32))
    mag = 20 * np.log(np.abs(fftshift(f)) + 1)

    # Normalise
    mag_norm = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# Noise Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def generate_noise_heatmap(gray: np.ndarray) -> Image.Image:
    """
    Generate a noise residual heatmap using Laplacian decomposition.

    Returns:
        PIL Image (RGB) showing noise distribution
    """
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.int16) - blur.astype(np.int16)
    residual_abs = np.abs(residual)
    norm = ((residual_abs - residual_abs.min()) /
            (residual_abs.max() - residual_abs.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# ELA Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def generate_ela_heatmap(ela_map: np.ndarray) -> Image.Image:
    """Convert the raw ELA diff map to a colour heatmap PIL Image."""
    if len(ela_map.shape) == 3:
        ela_gray = ela_map.mean(axis=2).astype(np.uint8)
    else:
        ela_gray = ela_map.astype(np.uint8)

    ela_gray = np.clip(ela_gray * 5, 0, 255).astype(np.uint8)
    heatmap  = cv2.applyColorMap(ela_gray, cv2.COLORMAP_PLASMA)
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

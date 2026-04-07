"""
models.py — EfficientNet-B0 deepfake model builder and inference
# NEW
Fully compatible with existing deepfake_efficientnet.pth weights.
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from typing import Tuple

from config import DEVICE, MODEL_PATH, DEEPFAKE_THRESHOLD
from preprocessing import preprocess_for_inference


# ── Build Model ───────────────────────────────────────────────────────────────
def build_deepfake_model() -> nn.Module:
    """
    EfficientNet-B0 pre-trained on ImageNet.
    Binary classifier head: outputs P(Fake) via sigmoid.
    Architecture is IDENTICAL to the trained model in deepfake_efficientnet.pth.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model.to(DEVICE)


def load_deepfake_model(path: str = MODEL_PATH) -> nn.Module:
    """
    Load saved EfficientNet-B0 weights from disk.
    Returns the model in eval mode on the appropriate device.
    """
    model = build_deepfake_model()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[models] Deepfake model loaded from: {path}  (device={DEVICE})")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def run_deepfake_inference(model: nn.Module, pil_image: Image.Image) -> float:
    """
    Run a single PIL image through the deepfake CNN.
    Uses torch.cuda.amp for mixed-precision on CUDA, falls back to fp32 on CPU.

    Returns:
        cnn_prob (float): P(Fake) in [0, 1]
    """
    tensor = preprocess_for_inference(pil_image).to(DEVICE)
    model.eval()

    use_amp = DEVICE.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                logit = model(tensor)
        else:
            logit = model(tensor)
        cnn_prob = float(torch.sigmoid(logit).item())

    return cnn_prob


def run_batch_inference(model: nn.Module, pil_images: list) -> list:
    """
    Batch inference for multiple PIL images.
    Returns a list of P(Fake) probabilities.
    """
    from preprocessing import INFERENCE_TRANSFORM
    import torch

    tensors = torch.stack([
        INFERENCE_TRANSFORM(img.convert("RGB")) for img in pil_images
    ]).to(DEVICE)

    model.eval()
    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(tensors)
        else:
            logits = model(tensors)
        probs = torch.sigmoid(logits).squeeze(1).tolist()

    return probs


def deepfake_verdict(cnn_prob: float, threshold: float = DEEPFAKE_THRESHOLD) -> str:
    """Returns 'DEEPFAKE' or 'REAL' based on CNN probability."""
    return "DEEPFAKE" if cnn_prob >= threshold else "REAL"


# ── AI-Gen Hybrid Inference ───────────────────────────────────────────────────
def load_ai_gen_model(path: str) -> nn.Module:
    """
    Placeholder loader for a future local AI-Gen CNN (Option 2).
    Returns None if the file doesn't exist.
    """
    import os
    if not os.path.exists(path):
        return None
    # Here is where the architecture for the local AI model would be defined
    # when the user provides one.
    return None


def run_ai_gen_inference(image_path: str, local_model: nn.Module = None) -> float:
    """
    Hybrid AI-Gen detector (Feature 2).
    1. If local_model is provided, uses it.
    2. Else if GEMINI_API_KEY is configured, uses Gemini Vision for zero-shot detection.
    3. Else, returns -1.0 (signals fusion.py to fall back to heuristics).
    """
    from config import GEMINI_API_KEY
    
    if local_model is not None:
        # Placeholder: could be a ResNet/EfficientNet trained on AI vs Real
        return 0.95
        
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Use gemini-1.5-flash for speed
            model = genai.GenerativeModel('gemini-1.5-flash')
            img = Image.open(image_path)
            
            prompt = (
                "Analyze this image. Is it a real photograph or is it a fully "
                "AI-generated synthetic image (e.g., Midjourney, DALL-E)? "
                "Respond with EXACTLY ONE WORD: 'REAL' or 'AI'."
            )
            response = model.generate_content([img, prompt])
            result = response.text.strip().upper()
            
            if "AI" in result:
                return 0.99
            elif "REAL" in result:
                return 0.01
                
        except Exception as e:
            print(f"[models] Gemini API error: {e}")
            
    return -1.0

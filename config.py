"""
config.py — Centralised configuration for the AI Forensic System
# NEW
"""

import os
import torch

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "deepfake_efficientnet.pth")
OUTPUT_DIR        = os.path.join(BASE_DIR, "output_reports")
PREPROCESSED_DIR  = os.path.join(BASE_DIR, "preprocessed images")
INPUT_DIR         = os.path.join(BASE_DIR, "input_images")

DATASET_BASE      = r"d:\Final Year Project\DeepFake Analysis\Dataset"
REAL_DIR          = os.path.join(DATASET_BASE, "Real")
FAKE_DIR          = os.path.join(DATASET_BASE, "Fake")          # For Deepfakes
AIGEN_DIR         = os.path.join(DATASET_BASE, "AI_Generated")  # For GAN/Diffusion

# ── Image / Preprocessing ─────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)           # Width x Height for CNN input
ELA_QUALITY = 90                  # JPEG quality for Error Level Analysis

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH            = "deepfake_efficientnet.pth"
AI_GEN_MODEL_PATH     = "ai_gen_detector.pth"   # Optional secondary CNN for AI-gen
PREPROCESS_TARGET     = "preprocessed images"

# ── API Keys ─────────────────────────────────────────────────────────────
# Set to your Gemini API key, or keep None to rely on fallback heuristic
GEMINI_API_KEY        = None

# ── Device and Settings ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Classification Thresholds ─────────────────────────────────────────────────
# Dynamic threshold: use 0.50 as a calibrated default; can be overridden by
# ROC-optimal threshold if computed during training.
DEEPFAKE_THRESHOLD    = 0.50   # P(Fake) >= this → DEEPFAKE
AI_GEN_THRESHOLD      = 0.60   # AI-gen heuristic score >= this → AI GENERATED

# Forensic signal weights for AI-gen heuristic scoring (Feature 2)
# These prioritize ELA and Noise as they are more robust in low-pass sync detection.
AI_GEN_WEIGHTS = {
    "fft_uniformity":    0.10,
    "ela_uniformity":    0.40,
    "noise_smoothness":  0.35,
    "no_metadata":       0.15,
}

# ── Training Hyperparameters ──────────────────────────────────────────────────
TRAIN_RATIO     = 0.80
EPOCHS          = 15
FINETUNE_EPOCHS = 8
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
FINETUNE_LR     = 5e-5
PREPROCESS_TARGET = 300

# ── Face Detection ────────────────────────────────────────────────────────────
FACE_PAD_RATIO   = 0.15     # Padding around detected face bounding box (Feature 1)
MIN_FACE_SIZE    = 40       # Minimum face size in pixels to consider
ALIGN_EYES       = True     # Enable eye alignment (Feature 1)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
GRADCAM_ALPHA    = 0.5      # Overlay transparency for Grad-CAM heatmap

# ── Supported Image Extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ── Grade Thresholds ──────────────────────────────────────────────────────────
# Mapping probabilities/heuristics to forensic grades (Feature 5)
GRADE_S_MAX   = 0.15   # cnn_prob < 0.15 → S (Authentic)
GRADE_A_MAX   = 0.45   # 0.15 <= cnn_prob < 0.45 → A (Realistic AI/Deepfake)
GRADE_B_MAX   = 0.75   # 0.45 <= cnn_prob < 0.75 → B (Detectable manipulation)
                        # cnn_prob >= 0.75 → C (Obvious fake)

# ── ImageNet Normalisation (Standard for EfficientNet) ────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Ensure critical directories exist
os.makedirs(OUTPUT_DIR,       exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "Fake"), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "Real"), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "AI_Generated"), exist_ok=True)

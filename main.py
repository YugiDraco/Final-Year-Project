"""
main.py — Advanced Multi-Modal AI Forensic System  v4.1
# UPDATED
Entry point for:
  • Batch 3-class training / fine-tuning workflow (Real, Deepfake, AI-Generated)
  • CLI single-image or folder analysis
  • Full modular pipeline (all modules imported from separate files)

Usage:
  python main.py                            # train or load + analyse dataset
  python main.py --input input_images/      # analyse a folder (batch mode)
  python main.py --input path/to/img.jpg    # analyse single image
"""

import os
import sys
import random
import json
import argparse
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models

# ── Import all new modules ────────────────────────────────────────────────────
from config import (
    DEVICE, MODEL_PATH, OUTPUT_DIR, PREPROCESSED_DIR,
    FAKE_DIR, REAL_DIR, AIGEN_DIR, DATASET_BASE,
    IMG_SIZE, BATCH_SIZE, EPOCHS, FINETUNE_EPOCHS,
    LEARNING_RATE, FINETUNE_LR, TRAIN_RATIO, PREPROCESS_TARGET,
    IMAGE_EXTENSIONS,
)
from preprocessing   import preprocess_cv2_image, INFERENCE_TRANSFORM, TRAIN_TRANSFORM
from models          import build_deepfake_model, load_deepfake_model, run_deepfake_inference, run_ai_gen_inference
from forensics       import run_full_forensics
from face_detection  import detect_faces, mtcnn_available
from fusion          import fuse_decisions, aggregate_face_results
from grading         import compute_grade, grade_to_emoji, verdict_to_emoji
from reporting       import save_report, save_batch_summary
from explainability  import generate_gradcam_overlay, generate_fft_heatmap

# ── Optional Gemini ───────────────────────────────────────────────────────────
_GEMINI_ENABLED = False
try:
    import google.generativeai as genai
    from config import GEMINI_API_KEY
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _GEMINI_ENABLED = True
except ImportError:
    pass


# ==========================================
# PHASE 1 — IMAGE PREPROCESSING
# ==========================================

class MultiClassDataset(Dataset):
    """PyTorch Dataset for the preprocessed images (3 classes)."""

    def __init__(self, manifest, augment=False):
        self.items     = manifest
        self.transform = TRAIN_TRANSFORM if augment else INFERENCE_TRANSFORM

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        img   = Image.open(entry["dst_path"]).convert("RGB")
        return self.transform(img), entry["true_label"]


def preprocess_images(total=PREPROCESS_TARGET):
    """
    Select `total` images (split between Real, Fake, AI_Generated), apply preprocessing,
    save to PREPROCESSED_DIR, and return a manifest list.
    """
    third = total // 3
    if third < 1: third = 1
    
    print("\n" + "=" * 65)
    print(" [PHASE 1] IMAGE PREPROCESSING  (Real, Deepfake, AI-Generated)")
    print("=" * 65)

    manifest = []
    classes = [("Real", 0, REAL_DIR), ("Fake", 1, FAKE_DIR), ("AI_Generated", 2, AIGEN_DIR)]
    
    for label_name, label_val, src_dir in classes:
        os.makedirs(os.path.join(PREPROCESSED_DIR, label_name), exist_ok=True)
        if not os.path.exists(src_dir):
            print(f"  [!] Skip {label_name}: directory not found.")
            continue
            
        all_files = [f for f in os.listdir(src_dir)
                     if f.lower().endswith(IMAGE_EXTENSIONS)]
        
        if not all_files:
            print(f"  [!] Skip {label_name}: no files found.")
            continue
            
        selected = random.sample(all_files, min(third, len(all_files)))
        print("  [>>] Preprocessing %d %s images ..." % (len(selected), label_name))

        for fname in selected:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(PREPROCESSED_DIR, label_name, fname)

            if not os.path.exists(dst_path):
                img_bgr = cv2.imread(src_path)
                if img_bgr is None:
                    continue
                img_bgr = preprocess_cv2_image(img_bgr)
                cv2.imwrite(dst_path, img_bgr)

            manifest.append({
                "src_path":   src_path,
                "dst_path":   dst_path,
                "true_label": label_val,
                "label_name": label_name,
            })

    print("  [OK] %d images saved to '%s'" % (len(manifest), PREPROCESSED_DIR))
    return manifest


# ==========================================
# PHASE 2a — CNN TRAINING
# ==========================================

def train_model(manifest):
    """
    Fine-tune EfficientNet-B0 on the preprocessed dataset.
    Note: Updated for 3 classes (Real=0, Fake=1, AI=2).
    """
    print("\n" + "=" * 65)
    print(" [PHASE 2a] CNN TRAINING  (EfficientNet-B0 3-class)")
    print("  Device : " + str(DEVICE))
    print("  Images : %d   Epochs : %d" % (len(manifest), EPOCHS))
    print("=" * 65)

    if not manifest: return None

    random.shuffle(manifest)
    split    = int(len(manifest) * TRAIN_RATIO)
    tr_items = manifest[:split]
    vl_items = manifest[split:]

    tr_set  = MultiClassDataset(tr_items, augment=True)
    val_set = MultiClassDataset(vl_items, augment=False)

    labels      = [e["true_label"] for e in tr_items]
    class_count = [labels.count(0), labels.count(1), labels.count(2)]
    weights     = [1.0 / (class_count[l] + 1e-6) for l in labels]
    sampler     = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    _nw       = 0 # Safe for Windows
    _pm       = (DEVICE.type == "cuda")
    tr_loader  = DataLoader(tr_set,  batch_size=min(BATCH_SIZE, len(tr_set)), sampler=sampler,
                            num_workers=_nw, pin_memory=_pm)
    val_loader = DataLoader(val_set, batch_size=min(BATCH_SIZE, len(val_set)), shuffle=False,
                            num_workers=_nw, pin_memory=_pm)

    # Note: build_deepfake_model needs to handle num_classes=3 or we use sigmoid + mapping
    # For now we use the existing binary model (Fake vs Real) but train on 3-way labels
    # to maintain compatibility, where label 2 (AI) is treated as 'Fake' for the CNN.
    model     = build_deepfake_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_total = 0
        for imgs, labels in tr_loader:
            # Treat both Deepfake(1) and AI_Gen(2) as 'Fake'(1.0) for the binary CNN
            binary_labels = labels.clone().float()
            binary_labels[binary_labels > 0] = 1.0
            
            imgs   = imgs.to(DEVICE, non_blocking=True)
            binary_labels = binary_labels.to(DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, binary_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_total += labels.size(0)
            tr_loss  += loss.item()

        print(f"  Epoch [{epoch:2d}/{EPOCHS}]  Loss: {tr_loss/len(tr_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    return model


def batch_analyse(input_path: str, model: nn.Module):
    """Analyse all images in a folder or a single image (Feature 8)."""
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
    else:
        print(f"  [ERROR] Input path not found: {input_path}")
        return

    print("\n" + "=" * 65)
    print(f" [BATCH] Analysing {len(image_paths)} image(s) from: {input_path}")
    print("=" * 65)

    results = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {os.path.basename(path)}")
        results.append(analyse_single_image(path, model))

    save_batch_summary(results)


def analyse_single_image(image_path: str, model: nn.Module) -> dict:
    """Run full multi-modal forensic pipeline on one image (Feature 3)."""
    result = {"filename": os.path.basename(image_path), "error": None}
    try:
        from preprocessing import load_and_preprocess
        img_pre, gray, pil_pre = load_and_preprocess(image_path)
        
        # Face Detection
        pil_orig = Image.open(image_path).convert("RGB")
        face_crops, face_info, used_full = detect_faces(pil_orig)

        # Multi-Branch Inference (Feature 2)
        cnn_prob  = run_deepfake_inference(model, pil_pre)
        forensics = run_full_forensics(img_pre, gray, image_path)
        ai_detector_prob = run_ai_gen_inference(image_path)

        # Decision Fusion (Feature 3)
        verdict, conf, signals, probs = fuse_decisions(
            cnn_prob         = cnn_prob,
            ai_gen_score     = forensics["ai_gen_score"],
            fft_peak         = forensics["fft_peak"],
            fft_mean         = forensics["fft_mean"],
            ela_avg          = forensics["ela_avg"],
            noise_sigma      = forensics["noise_sigma"],
            editing_flag     = forensics["editing_flag"],
            ai_detector_prob = ai_detector_prob,
        )

        # Grading (Feature 5)
        grade, grade_desc, grade_reason = compute_grade(
            verdict      = verdict,
            cnn_prob     = cnn_prob,
            ai_gen_score = forensics["ai_gen_score"],
            fft_peak     = forensics["fft_peak"],
            fft_mean     = forensics["fft_mean"],
            ela_avg      = forensics["ela_avg"],
            noise_sigma  = forensics["noise_sigma"],
            confidence   = conf,
        )

        # JSON Reporting (Feature 7)
        save_report(
            image_path     = image_path,
            verdict        = verdict,
            confidence     = conf,
            grade          = grade,
            grade_desc     = grade_desc,
            grade_reason   = grade_reason,
            cnn_prob       = cnn_prob,
            ai_gen_score   = forensics["ai_gen_score"],
            forensics      = forensics,
            face_info      = face_info,
            per_face       = [],
            fusion_signals = signals,
        )

        print(f"  [RESULT] {verdict:14s} | Grade: {grade} | Conf: {conf*100:5.1f}%")

        result.update({
            "verdict":    verdict,
            "confidence": conf,
            "grade":      grade,
            "cnn_prob":   cnn_prob,
        })

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] {os.path.basename(image_path)}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Advanced AI Forensic Engine v4.1")
    parser.add_argument("--input", "-i", type=str, help="Path to image or folder for batch analysis")
    parser.add_argument("--train", "-t", action="store_true", help="Run training pipeline")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("\n" + "=" * 65)
    print("  AI FORENSIC SYSTEM (v4.1) — Production Authenticity Engine")
    print("=" * 65)

    # CLI Operation
    if args.input:
        if not os.path.exists(MODEL_PATH):
            print(f"  [!] Model not found at {MODEL_PATH}. Exit.")
            sys.exit(1)
        model = load_deepfake_model(MODEL_PATH)
        batch_analyse(args.input, model)
        return

    # Training Workflow
    if args.train:
        manifest = preprocess_images(PREPROCESS_TARGET)
        train_model(manifest)
        print(f"\n[DONE] Training complete. Model saved to {MODEL_PATH}")
        return

    # Default: Sample analysis if dataset exists
    if os.path.exists(MODEL_PATH):
        model = load_deepfake_model(MODEL_PATH)
        if os.path.exists(PREPROCESSED_DIR):
            batch_analyse(PREPROCESSED_DIR, model)
        else:
            print("  [INFO] No input provided. Launch UI via 'python app.py' for interactive use.")
    else:
        print("  [!] System uninitialized. Run with '--train' to build model.")


if __name__ == "__main__":
    main()
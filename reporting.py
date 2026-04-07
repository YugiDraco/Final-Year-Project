"""
reporting.py — Extended JSON report generation and batch processing
# NEW
Saves per-image forensic reports with face-level scores and fusion reasoning.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from config import OUTPUT_DIR


def save_report(
    image_path:     str,
    verdict:        str,
    confidence:     float,
    grade:          str,
    grade_desc:     str,
    grade_reason:   str,
    cnn_prob:       float,
    ai_gen_score:   float,
    forensics:      dict,
    face_info:      list,
    per_face:       list,
    fusion_signals: dict,
    output_dir:     str = OUTPUT_DIR,
) -> str:
    """
    Save a comprehensive JSON report for one processed image.

    Returns the path to the saved report file.
    """
    filename = os.path.basename(image_path)
    stem     = os.path.splitext(filename)[0]

    report = {
        "system":        "Advanced Multi-Modal AI Forensic System v4.0",
        "filename":      filename,
        "image_path":    image_path,
        "analysis_date": datetime.now().isoformat(),

        # ── Final Decision ─────────────────────────────────────────────────
        "verdict": {
            "classification": verdict,             # REAL | DEEPFAKE | AI GENERATED
            "confidence_%":   round(confidence * 100, 2),
            "grade":          grade,               # S | A | B | C
            "grade_description": grade_desc,
            "grade_reason":   grade_reason,
        },

        # ── CNN Signal ────────────────────────────────────────────────────
        "cnn_analysis": {
            "model":          "EfficientNet-B0 (ImageNet → Deepfake fine-tuned)",
            "fake_probability": round(cnn_prob, 4),
            "confidence_%":   round(cnn_prob * 100, 2),
        },

        # ── AI Generated Signal ───────────────────────────────────────────
        "ai_gen_analysis": {
            "heuristic_score": round(ai_gen_score, 4),
            "method": "Weighted forensic signals (FFT + ELA + Noise + EXIF)",
        },

        # ── Forensic Metrics ──────────────────────────────────────────────
        "forensic_metrics": {
            "ela_avg_diff":   round(forensics.get("ela_avg", 0), 3),
            "noise_sigma":    round(forensics.get("noise_sigma", 0), 2),
            "fft_peak":       round(forensics.get("fft_peak", 0), 2),
            "fft_mean":       round(forensics.get("fft_mean", 0), 2),
            "fft_ratio":      round(
                forensics.get("fft_peak", 0) / (forensics.get("fft_mean", 1) + 1e-6), 2
            ),
            "editing_flag":   forensics.get("editing_flag", False),
            "has_exif":       forensics.get("has_exif", False),
        },

        # ── Face Detection ────────────────────────────────────────────────
        "face_detection": {
            "face_count":    len(face_info),
            "used_fullimage": all(f.get("box") is None for f in face_info),
            "faces":         face_info,
        },

        # ── Per-Face Analysis ─────────────────────────────────────────────
        "per_face_analysis": per_face,

        # ── Fusion Reasoning ──────────────────────────────────────────────
        "fusion_reasoning": fusion_signals,

        # ── Preprocessing ─────────────────────────────────────────────────
        "preprocessing": {
            "resize":       "224x224",
            "denoise":      "Gaussian Blur σ=0.8",
            "enhancement":  "CLAHE on LAB L-channel",
            "normalisation":"ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        },
    }

    report_path = os.path.join(output_dir, f"report_{stem}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, ensure_ascii=False)

    return report_path


def save_batch_summary(results: List[Dict], output_dir: str = OUTPUT_DIR) -> str:
    """
    Save a batch-level summary JSON after processing a folder of images.
    Includes aggregate metrics and verdict distribution.
    """
    total = len(results)
    if total == 0:
        return ""

    verdict_counts = {"REAL": 0, "DEEPFAKE": 0, "AI GENERATED": 0}
    grade_counts   = {"S": 0, "A": 0, "B": 0, "C": 0}
    errors         = 0

    for r in results:
        v = r.get("verdict", "REAL")
        g = r.get("grade",   "S")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        grade_counts[g]   = grade_counts.get(g, 0) + 1
        if r.get("error"):
            errors += 1

    summary = {
        "system":         "Advanced Multi-Modal AI Forensic System v4.0",
        "batch_timestamp": datetime.now().isoformat(),
        "total_images":    total,
        "processed":       total - errors,
        "errors":          errors,
        "verdict_distribution": verdict_counts,
        "grade_distribution":   grade_counts,
        "results":         results,
    }

    path = os.path.join(output_dir, "batch_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4, ensure_ascii=False)

    return path

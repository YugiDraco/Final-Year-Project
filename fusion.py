"""
fusion.py — Decision Fusion Engine
# NEW
Combines EfficientNet deepfake probability + AI-gen heuristic score
+ forensic signals to produce the final 3-class verdict:
    REAL | DEEPFAKE | AI GENERATED
"""

from typing import Tuple, Dict
from config import DEEPFAKE_THRESHOLD, AI_GEN_THRESHOLD


def fuse_decisions(
    cnn_prob:       float,
    ai_gen_score:   float,
    fft_peak:       float,
    fft_mean:       float,
    ela_avg:        float,
    noise_sigma:    float,
    editing_flag:   bool,
    ai_detector_prob: float = -1.0,
) -> Tuple[str, float, Dict, Dict[str, float]]:
    """
    Multi-signal decision fusion.

    Decision Logic (priority order):
        1. If cnn_prob >= DEEPFAKE_THRESHOLD        → DEEPFAKE
           (EfficientNet is our most reliable signal for manipulated real faces)
        2. Elif ai_gen_score >= AI_GEN_THRESHOLD    → AI GENERATED
           (Heuristic signals suggest fully synthetic image)
        3. Else                                      → REAL

    Confidence is computed as a calibrated weighted blend of the signals.

    Returns:
        verdict     (str):  "REAL" | "DEEPFAKE" | "AI GENERATED"
        confidence  (float): [0, 1] confidence in the verdict
        signals     (dict): per-signal breakdown for transparency
        probs       (dict): 3-class probabilities for UI bars
    """
    fft_ratio = fft_peak / (fft_mean + 1e-6)

    signals = {
        "cnn_fake_prob":    round(cnn_prob, 4),
        "ai_gen_score":     round(ai_gen_score, 4),
        "fft_ratio":        round(fft_ratio, 2),
        "ela_avg":          round(ela_avg, 3),
        "noise_sigma":      round(noise_sigma, 2),
        "editing_flag":     editing_flag,
        "deepfake_threshold": DEEPFAKE_THRESHOLD,
        "ai_gen_threshold":   AI_GEN_THRESHOLD,
    }

    # ── Step 1: Calculate Probabilities (The 3-Way Distribution) ─────────────
    probs = _compute_3way_probs(cnn_prob, ai_gen_score, ai_detector_prob)
    
    # ── Step 2: Apply Dominance Rule ──────────────────────────────────────────
    # The category with the highest probability becomes the verdict.
    verdict = max(probs, key=probs.get)
    
    # ── Step 3: Compute Confidence & Reason ───────────────────────────────────
    if verdict == "DEEPFAKE":
        confidence = _calibrate_confidence(cnn_prob, DEEPFAKE_THRESHOLD, 1.0)
        signals["primary_signal"] = "CNN Dominance (P=%.2f%%)" % (probs["DEEPFAKE"] * 100)
        signals["fusion_reason"]  = (
            f"CNN model identifies manipulation (P(Fake)={cnn_prob:.4f}). "
            f"This category is dominant in the probability distribution."
        )
        
    elif verdict == "AI GENERATED":
        score_for_conf = ai_detector_prob if ai_detector_prob != -1.0 else ai_gen_score
        thresh_for_conf = 0.5 if ai_detector_prob != -1.0 else AI_GEN_THRESHOLD
        
        confidence = _calibrate_confidence(score_for_conf, thresh_for_conf, 1.0)
        signals["primary_signal"] = "AI-Gen Dominance (P=%.2f%%)" % (probs["AI GENERATED"] * 100)
        signals["fusion_reason"]  = (
            f"Synthetic patterns identified (Score={score_for_conf:.4f}). "
            f"Category consistent with non-biological synthesis."
        )
        
    else:  # REAL
        # Confidence logic for Real: gap between Real and the next best "fake" category
        next_best = max(probs["DEEPFAKE"], probs["AI GENERATED"])
        # We use raw scores to calibrate how "Real" it is
        real_conf_cnn   = 1.0 - (cnn_prob / (DEEPFAKE_THRESHOLD + 1e-9))
        real_conf_aigen = 1.0 - (ai_gen_score / (AI_GEN_THRESHOLD + 1e-9))
        confidence = 0.5 + 0.5 * (0.6 * real_conf_cnn + 0.4 * real_conf_aigen)
        confidence = float(max(0.5, min(1.0, confidence)))
        
        signals["primary_signal"] = "Authenticity Dominance (P=%.2f%%)" % (probs["REAL"] * 100)
        signals["fusion_reason"]  = (
            "No significant manipulation or synthesis traces were found. "
            "Probability of authenticity exceeds all suspect categories."
        )

    return verdict, round(confidence, 4), signals, probs


def _calibrate_confidence(score: float, low: float, high: float) -> float:
    """Linearly scale score from [low, high] → [0.5, 1.0]."""
    if high <= low:
        return 0.5
    normalised = (score - low) / (high - low)
    return float(max(0.5, min(1.0, 0.5 + 0.5 * normalised)))


def _compute_3way_probs(cnn_prob, ai_gen_score, ai_detector_prob) -> Dict[str, float]:
    """
    Estimates 3-way probabilities based on the various detector signals.
    Applies logic to allow for competition between categories.
    """
    # 1. Deepfake probability (from face-swap CNN)
    p_deepfake = float(cnn_prob)

    # 2. AI-Generated probability
    if ai_detector_prob != -1.0:
        # Weighted blend: give more weight to explicit AI detectors
        p_ai_gen = (0.75 * ai_detector_prob) + (0.25 * ai_gen_score)
    else:
        p_ai_gen = float(ai_gen_score)

    # 3. Real probability (the residual / ground truth)
    # Total suspicion is the maximum of the two fake signals
    total_suspicion = max(p_deepfake, p_ai_gen)
    p_real = max(0.0, 1.0 - total_suspicion)

    # ── Soft-Competition Alignment ──────────────────────────────────────────
    # If a signal exceeds its HIGH-risk threshold, boost it to ensure dominance.
    if p_deepfake > DEEPFAKE_THRESHOLD:
        p_deepfake *= 1.2
    if p_ai_gen > AI_GEN_THRESHOLD:
        p_ai_gen *= 1.2

    # Re-normalize to sum to 1.0
    s = p_real + p_deepfake + p_ai_gen + 1e-9
    return {
        "REAL":          round(p_real / s, 4),
        "DEEPFAKE":      round(p_deepfake / s, 4),
        "AI GENERATED":  round(p_ai_gen / s, 4),
    }


def aggregate_face_results(face_results: list) -> dict:
    """
    Aggregate per-face analysis results for multi-face images.

    Strategy:
        - Final verdict = verdict of face with highest risk score
        - Max deepfake prob = worst-case face
        - Confidence = max confidence across all faces

    Args:
        face_results: list of dicts, each with keys:
            verdict, confidence, cnn_prob, ai_gen_score, signals, face_id

    Returns:
        aggregated dict with final verdict and summary
    """
    if not face_results:
        return {"verdict": "REAL", "confidence": 0.5, "face_count": 0}

    # Risk priority: DEEPFAKE > AI GENERATED > REAL
    risk_rank = {"DEEPFAKE": 2, "AI GENERATED": 1, "REAL": 0}
    face_results_sorted = sorted(
        face_results,
        key=lambda x: (risk_rank.get(x["verdict"], 0), x["confidence"]),
        reverse=True
    )

    worst = face_results_sorted[0]

    return {
        "verdict":           worst["verdict"],
        "confidence":        worst["confidence"],
        "face_count":        len(face_results),
        "dominant_face_id":  worst.get("face_id", 0),
        "max_cnn_prob":      max(f.get("cnn_prob", 0) for f in face_results),
        "max_ai_gen_score":  max(f.get("ai_gen_score", 0) for f in face_results),
        "per_face":          face_results,
        "fusion_reason":     worst.get("signals", {}).get("fusion_reason", ""),
    }

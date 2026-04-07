"""
grading.py — Smart grading system using multi-signal fusion
# NEW
Grades: S (Authentic) → A (Highly Realistic) → B (Detectable) → C (Obvious Fake)
"""

from typing import Tuple
from config import GRADE_S_MAX, GRADE_A_MAX, GRADE_B_MAX


GRADE_DESCRIPTIONS = {
    "S": "Confirmed Authentic",
    "A": "Highly Realistic AI / Deepfake",
    "B": "Detectable Manipulation",
    "C": "Obvious Fake",
}


def compute_grade(
    verdict:       str,
    cnn_prob:      float,
    ai_gen_score:  float,
    fft_peak:      float,
    fft_mean:      float,
    ela_avg:       float,
    noise_sigma:   float,
    confidence:    float,
) -> Tuple[str, str, str]:
    """
    Compute a forensic grade (S / A / B / C) based on multi-signal dominance (Feature 5).
    """
    fft_ratio = fft_peak / (fft_mean + 1e-6)
    
    # Use the strongest suspicion signal as the grading anchor
    max_suspicion = max(cnn_prob, ai_gen_score)

    # ── Grade S: Authentic ───────────────────────────────────────────────
    if verdict == "REAL" and max_suspicion < GRADE_S_MAX:
        grade  = "S"
        reason = (
            f"Media appears authentic. Forensic signals (P_Fake={cnn_prob:.4f}, "
            f"AI_Score={ai_gen_score:.4f}) are within natural bounds. "
            f"No anomalies detected. Confidence: {confidence*100:.1f}%."
        )

    # ── Grade A: High-Quality AI / Deepfake ──────────────────────────────
    elif max_suspicion < GRADE_A_MAX:
        grade  = "A"
        if verdict == "AI GENERATED":
            reason = (
                f"Highly realistic synthetic media. AI-gen score ({ai_gen_score:.4f}) "
                f"and spectral patterns (FFT ratio={fft_ratio:.2f}) suggest "
                f"high-fidelity non-biological origin. Confidence: {confidence*100:.1f}%."
            )
        else:
            reason = (
                f"High-fidelity deepfake. CNN detection probability ({cnn_prob:.4f}) "
                f"indicates professional-grade GAN manipulation. Confidence: {confidence*100:.1f}%."
            )

    # ── Grade B: Detectable Manipulation ──────────────────────────────────
    elif max_suspicion < GRADE_B_MAX:
        grade  = "B"
        if verdict == "AI GENERATED":
            reason = (
                f"Detectable AI synthesis. Heuristic score ({ai_gen_score:.4f}) and "
                f"noise residual ({noise_sigma:.2f}) reveal clear artificial signatures. "
                "Confidence: %s%%." % round(confidence*100, 1)
            )
        else:
            reason = (
                "Detectable face manipulation. CNN probability is %.4f. "
                "FFT and ELA analysis confirm spectral artifacts. "
                "Confidence: %s%%." % (cnn_prob, round(confidence*100, 1))
            )

    # ── Grade C: Obvious Manipulation ─────────────────────────────────────
    else:
        grade  = "C"
        reason = (
            f"Critical manipulation detected. Strong suspicion signal ({max_suspicion:.4f}) "
            f"combined with inconsistent noise residuals presents a definitive "
            f"signature of AI generation or face swapping. Confidence: {confidence*100:.1f}%."
        )

    return grade, GRADE_DESCRIPTIONS[grade], reason


def grade_to_emoji(grade: str) -> str:
    """Return a coloured emoji indicator for a grade."""
    return {
        "S": "🟢",   # Green — authentic
        "A": "🟡",   # Yellow — realistic fake
        "B": "🟠",   # Orange — detectable
        "C": "🔴",   # Red — obvious fake
    }.get(grade, "⚪")


def verdict_to_emoji(verdict: str) -> str:
    """Return an emoji for a verdict string."""
    return {
        "REAL":         "✅",
        "DEEPFAKE":     "🚨",
        "AI GENERATED": "🤖",
    }.get(verdict, "❓")

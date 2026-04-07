"""
app.py — Advanced Multi-Modal AI Forensic System  v4.0
# UPDATED
Gradio web UI with:
  • Face detection (MTCNN)
  • 3-class verdict: REAL | DEEPFAKE | AI GENERATED
  • Grad-CAM heatmap overlay
  • FFT frequency heatmap
  • Noise residual map
  • Full forensic metrics panel
  • Extended JSON report
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from PIL import Image

# Ensure the project folder is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config       import MODEL_PATH, DEVICE, OUTPUT_DIR, DEEPFAKE_THRESHOLD, AI_GEN_THRESHOLD
from preprocessing import load_and_preprocess, preprocess_for_inference
from face_detection import detect_faces, draw_face_boxes, mtcnn_available
from models       import load_deepfake_model, run_deepfake_inference, run_ai_gen_inference
from forensics    import run_full_forensics
from fusion       import fuse_decisions, aggregate_face_results
from explainability import generate_gradcam_overlay, generate_fft_heatmap, generate_noise_heatmap
from grading      import compute_grade, grade_to_emoji, verdict_to_emoji
from reporting    import save_report

# ── Load model at startup ─────────────────────────────────────────────────────
print("=" * 60)
print("  Advanced Multi-Modal AI Forensic System  v4.0")
print(f"  Device: {DEVICE}")
print("=" * 60)

_model = None
_model_error = None
try:
    _model = load_deepfake_model(MODEL_PATH)
    print("  [OK] Deepfake model loaded successfully.")
except Exception as e:
    _model_error = str(e)
    print(f"  [ERROR] Model load failed: {e}")

print(f"  [INFO] MTCNN face detection: {'enabled' if mtcnn_available() else 'unavailable (fallback to full image)'}")
print("=" * 60)


# ── Core analysis pipeline ────────────────────────────────────────────────────
def analyze_image(image_filepath):
    """
    Full forensic analysis pipeline:
      1. Load + preprocess
      2. Face detection
      3. Per-face CNN inference + forensics
      4. Decision fusion
      5. Explainability (Grad-CAM, FFT, Noise)
      6. Grading + report
    """
    # Sentinel returns for all outputs
    ERROR_TUPLE = (None, None, None, None, "⚠️ Error", "", "", "", "")

    if image_filepath is None:
        return (None, None, None, None,
                "⚠️ Please upload an image.",
                "", "", "", "")

    if _model is None:
        return (None, None, None, None,
                f"❌ Model not loaded: {_model_error}",
                "", "", "", "")

    try:
        # ── Step 1: Load ──────────────────────────────────────────────────
        pil_orig = Image.open(image_filepath).convert("RGB")
        img_bgr, gray, pil_pre = load_and_preprocess(image_filepath)

        # ── Step 2: Face Detection ────────────────────────────────────────
        face_crops, face_info, used_full = detect_faces(pil_orig)
        face_annotated = draw_face_boxes(pil_orig, face_info)

        # ── Step 3: Forensic extraction + CNN ──────────────────────────────
        # IMPORTANT: Forensics MUST run on strictly unmodified RAW pixels.
        raw_bgr = cv2.imread(image_filepath)
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        forensics = run_full_forensics(raw_bgr, raw_gray, image_filepath)

        # IMPORTANT: CNN must run on the full preprocessed image (resize+denoise+
        # CLAHE+normalize) — it was trained on full images, NOT face crops.
        # Face crops are used ONLY for visualisation bounding boxes.

        # Run CNN on the full preprocessed image — consistent with training
        cnn_prob = run_deepfake_inference(_model, pil_pre)

        # Run Hybrid AI Detector (API / Local CNN)
        ai_detector_prob = run_ai_gen_inference(image_filepath, local_model=None)

        # Fusion (single image-level decision)
        verdict_f, conf_f, signals_f, probs_f = fuse_decisions(
            cnn_prob         = cnn_prob,
            ai_gen_score     = forensics["ai_gen_score"],
            fft_peak         = forensics["fft_peak"],
            fft_mean         = forensics["fft_mean"],
            ela_avg          = forensics["ela_avg"],
            noise_sigma      = forensics["noise_sigma"],
            editing_flag     = forensics["editing_flag"],
            ai_detector_prob = ai_detector_prob,
        )

        per_face_results = [{
            "face_id":      0,
            "verdict":      verdict_f,
            "confidence":   conf_f,
            "cnn_prob":     cnn_prob,
            "ai_gen_score": forensics["ai_gen_score"],
            "signals":      signals_f,
            "probs":        probs_f,
        }]

        # ── Step 4: Aggregate multi-face results ──────────────────────────
        agg = aggregate_face_results(per_face_results)
        final_verdict    = agg["verdict"]
        final_confidence = agg["confidence"]
        final_cnn_prob   = agg["max_cnn_prob"]
        final_ai_score   = agg["max_ai_gen_score"]
        final_probs      = per_face_results[0]["probs"]  # for UI chart

        # Re-run forensics for grading signals (use the aggregated forensics)
        forensics = run_full_forensics(img_bgr, gray, image_filepath)

        # ── Step 5: Grading ───────────────────────────────────────────────
        grade, grade_desc, grade_reason = compute_grade(
            verdict      = final_verdict,
            cnn_prob     = final_cnn_prob,
            ai_gen_score = final_ai_score,
            fft_peak     = forensics["fft_peak"],
            fft_mean     = forensics["fft_mean"],
            ela_avg      = forensics["ela_avg"],
            noise_sigma  = forensics["noise_sigma"],
            confidence   = final_confidence,
        )

        # Re-run explainability on RAW images for pure signal visualization
        gradcam_img = generate_gradcam_overlay(_model, pil_orig)
        fft_img     = generate_fft_heatmap(raw_gray)
        noise_img   = generate_noise_heatmap(raw_gray)

        # ── Step 7: Save JSON Report ──────────────────────────────────────
        try:
            save_report(
                image_path     = image_filepath,
                verdict        = final_verdict,
                confidence     = final_confidence,
                grade          = grade,
                grade_desc     = grade_desc,
                grade_reason   = grade_reason,
                cnn_prob       = final_cnn_prob,
                ai_gen_score   = final_ai_score,
                forensics      = forensics,
                face_info      = face_info,
                per_face       = per_face_results,
                fusion_signals = agg,
            )
        except Exception:
            pass   # Don't let report saving break the UI

        # ── Step 8: Format UI outputs ─────────────────────────────────────
        v_emoji = verdict_to_emoji(final_verdict)
        g_emoji = grade_to_emoji(grade)

        verdict_text = (
            f"{v_emoji}  FINAL VERDICT:  {final_verdict}\n"
            f"   Confidence:    {final_confidence*100:.1f}%\n"
            f"   Faces Detected: {agg['face_count']}  "
            f"{'(full image fallback)' if used_full else ''}"
        )

        grade_text = (
            f"{g_emoji}  Grade {grade} — {grade_desc}\n\n"
            f"{grade_reason}"
        )

        fft_ratio = forensics["fft_peak"] / (forensics["fft_mean"] + 1e-6)
        metrics_text = (
            f"{'─'*42}\n"
            f"  VERDICT CLASS     :  {final_verdict}\n"
            f"  DOMINANT PROB     :  {final_probs[final_verdict]*100:>6.2f}%\n"
            f"{'─'*42}\n"
            f"  CNN P(Fake)       :  {final_cnn_prob*100:>6.2f}%\n"
            f"  AI-Gen Score      :  {final_ai_score:>6.4f}\n"
            f"{'─'*42}\n"
            f"  ELA Avg Diff      :  {forensics['ela_avg']:>6.3f}\n"
            f"  Noise σ           :  {forensics['noise_sigma']:>6.2f}\n"
            f"  FFT Peak          :  {forensics['fft_peak']:>6.2f}\n"
            f"  FFT Mean          :  {forensics['fft_mean']:>6.2f}\n"
            f"  FFT Ratio         :  {fft_ratio:>6.2f}\n"
            f"{'─'*42}\n"
            f"  Editing Flag      :  {forensics['editing_flag']}\n"
            f"  EXIF Present      :  {forensics['has_exif']}\n"
            f"  MTCNN Available   :  {mtcnn_available()}\n"
            f"{'─'*42}\n"
            f"  Deepfake Thresh   :  {DEEPFAKE_THRESHOLD}\n"
            f"  AI-Gen Thresh     :  {AI_GEN_THRESHOLD}\n"
        )

        fusion_text = agg.get("fusion_reason", "N/A")

        return (
            face_annotated,     # out_faces
            gradcam_img,        # out_gradcam
            fft_img,            # out_fft
            noise_img,          # out_noise
            verdict_text,       # out_verdict
            grade_text,         # out_grade
            metrics_text,       # out_metrics
            fusion_text,        # out_fusion
            final_probs,        # out_label_chart
        )

    except Exception as exc:
        import traceback
        err = f"❌ Error during analysis:\n{exc}\n\n{traceback.format_exc()}"
        return (None, None, None, None, err, "", "", "")


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', -apple-system, sans-serif; background-color: #f9fafb; }
.verdict-box textarea { font-size:1.2em !important; font-weight:700 !important; color: #111827 !important; }
.grade-box textarea { font-size:1.0em !important; line-height: 1.5 !important; }
.metric-box textarea { font-family: 'Fira Code', monospace !important; font-size: 0.9em !important; background: #f3f4f6 !important; }
#title-container { text-align: center; margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; border-radius: 12px; }
#title-container h1 { margin: 0; font-size: 2.2rem; letter-spacing: -0.025em; }
.stat-card { background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
"""

with gr.Blocks(
    title="AI Forensic System v4.1",
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🔬 Advanced Multi-Modal AI Forensic System
        Unified Authenticity Engine: REAL | DEEPFAKE | AI GENERATED
        """,
        elem_id="title-container"
    )

    # ── Grading Legend ────────────────────────────────────────────────────────
    with gr.Row():
        gr.Markdown("""
        <div style='display: flex; justify-content: space-around; width: 100%; font-size: 0.9rem;'>
            <div class='stat-card'>🟢 <b>Grade S</b>: Authentic</div>
            <div class='stat-card'>🟡 <b>Grade A</b>: Realistic AI</div>
            <div class='stat-card'>🟠 <b>Grade B</b>: Manipulated</div>
            <div class='stat-card'>🔴 <b>Grade C</b>: Obvious Fake</div>
        </div>
        """)

    with gr.Row(variant="panel"):
        # ── Left Column: Controls & Primary Results ───────────────────────────
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="📷 Upload Evidence", height=320)
            analyze_btn = gr.Button("🔎 Run Forensic Analysis", variant="primary", size="lg")
            
            with gr.Group():
                out_verdict = gr.Textbox(label="🏁 Final Verdict", lines=2, interactive=False, elem_classes=["verdict-box"])
                out_grade = gr.Textbox(label="📊 Forensic Grade", lines=4, interactive=False, elem_classes=["grade-box"])
            
            out_label_chart = gr.Label(label="📈 Probability Distribution", num_top_classes=3)

        # ── Right Column: Analysis Dossier ────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🖼️ Visual Evidence"):
                    with gr.Row():
                        out_faces   = gr.Image(label="👁️ Detected & Aligned Faces", height=250)
                        out_gradcam = gr.Image(label="🔥 Attention Heatmap (Grad-CAM)", height=250)
                    with gr.Row():
                        out_fft     = gr.Image(label="📡 Spectral Map (FFT)", height=250)
                        out_noise   = gr.Image(label="🌫️ Noise Residual", height=250)
                
                with gr.TabItem("📊 Forensic Metrics"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            out_metrics = gr.Textbox(label="Raw Signal Data", lines=15, interactive=False, elem_classes=["metric-box"])
                        with gr.Column(scale=1):
                            out_fusion = gr.Textbox(label="🧠 Decision Logic", lines=15, interactive=False)

    # ── Wire up ───────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_img],
        outputs=[
            out_faces, out_gradcam, out_fft, out_noise,
            out_verdict, out_grade, out_metrics, out_fusion, out_label_chart
        ]
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #6b7280; font-size: 0.8rem;'>
        <b>Engine v4.1</b> | EfficientNet-B0 + MTCNN Alignment + Hybrid AI-Gen Heuristics | 
        Reports archived to <code>output_reports/</code>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        inbrowser=True,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    )

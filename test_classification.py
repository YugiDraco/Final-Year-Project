import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion import fuse_decisions

def test_case(name, cnn, ai_score, ai_api=-1.0):
    print(f"\n--- Test: {name} ---")
    print(f"Inputs: CNN={cnn}, AI-Gen Score={ai_score}, AI-API={ai_api}")
    
    verdict, conf, signals, probs = fuse_decisions(
        cnn_prob=cnn,
        ai_gen_score=ai_score,
        fft_peak=10.0,
        fft_mean=1.0,
        ela_avg=0.05,
        noise_sigma=2.0,
        editing_flag=False,
        ai_detector_prob=ai_api
    )
    
    print(f"VERDICT: {verdict}")
    print(f"CONFIDENCE: {conf*100:.1f}%")
    print(f"PROBS: {probs}")
    print(f"REASON: {signals['fusion_reason']}")
    return verdict

if __name__ == "__main__":
    # Case 1: Clear Deepfake
    assert test_case("Clear Deepfake", 0.9, 0.2) == "DEEPFAKE"
    
    # Case 2: Clear AI Generated
    assert test_case("Clear AI Gen", 0.1, 0.8) == "AI GENERATED"
    
    # Case 3: Mixed (Dominant AI Gen)
    # Previously this might have been DEEPFAKE due to priority
    assert test_case("Mixed Signal (AI Dominant)", 0.55, 0.8) == "AI GENERATED"
    
    # Case 4: Mixed (Dominant CNN)
    assert test_case("Mixed Signal (CNN Dominant)", 0.8, 0.55) == "DEEPFAKE"
    
    # Case 5: Real
    assert test_case("Clear Real", 0.1, 0.1) == "REAL"
    
    # Case 6: API Overrides
    assert test_case("API Confirmation", 0.2, 0.3, 0.9) == "AI GENERATED"

    print("\n[SUCCESS] All classification test cases passed.")

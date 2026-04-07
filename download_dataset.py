"""
download_dataset.py
Downloads the OpenFake dataset from HuggingFace:
  ComplexDataLab/OpenFake  — Real, Deepfake, AI-Generated images
Organises images into:
  d:/Final Year Project/DeepFake Analysis/Dataset/
      ├── Real/
      ├── Fake/         (Deepfakes)
      └── AI_Generated/
"""

import os
import sys
import io

BASE_DIR  = r"d:\Final Year Project\DeepFake Analysis\Dataset"
REAL_DIR  = os.path.join(BASE_DIR, "Real")
FAKE_DIR  = os.path.join(BASE_DIR, "Fake")
AIGEN_DIR = os.path.join(BASE_DIR, "AI_Generated")

for d in (REAL_DIR, FAKE_DIR, AIGEN_DIR):
    os.makedirs(d, exist_ok=True)

try:
    from datasets import load_dataset
except ImportError:
    print("[INFO] Installing HuggingFace 'datasets' library …")
    os.system(f"{sys.executable} -m pip install datasets pillow -q")
    from datasets import load_dataset

from PIL import Image

print("=" * 60)
print("  Downloading ComplexDataLab/OpenFake from HuggingFace")
print("=" * 60)

# Stream to avoid huge RAM usage — downloads one batch at a time
ds = load_dataset(
    "ComplexDataLab/OpenFake",
    split="train",
    streaming=True,
)

label_map = {
    "real":          REAL_DIR,
    "deepfake":      FAKE_DIR,
    "fake":          FAKE_DIR,
    "ai_generated":  AIGEN_DIR,
    "ai generated":  AIGEN_DIR,
    "ai-generated":  AIGEN_DIR,
    "synthetic":     AIGEN_DIR,
}

counters = {"Real": 0, "Fake": 0, "AI_Generated": 0}
MAX_PER_CLASS = 5000   # cap per-class — remove this line for full download

for i, sample in enumerate(ds):
    try:
        # OpenFake classification:
        # label: 'real' or 'fake'
        # type: 'swap', 'gan', 'diffusion', 'etc'
        raw_label = str(sample.get("label", "")).strip().lower()
        raw_type  = str(sample.get("type", "")).strip().lower()

        if raw_label == "real":
            out_dir = REAL_DIR
        elif raw_label == "fake":
            if "swap" in raw_type:
                out_dir = FAKE_DIR      # "Deepfake" (face swapped)
            else:
                out_dir = AIGEN_DIR     # "AI Generated" (fully synthetic)
        else:
            continue   # unknown label

        class_key = os.path.basename(out_dir)
        if counters.get(class_key, 0) >= MAX_PER_CLASS:
            continue

        # Get image (PIL or raw bytes)
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img)).convert("RGB")
        else:
            img = img.convert("RGB")

        fname = f"{class_key.lower()}_{i:07d}.jpg"
        img.save(os.path.join(out_dir, fname), "JPEG", quality=95)
        counters[class_key] = counters.get(class_key, 0) + 1

        if i % 500 == 0:
            print(f"  [{i:>7}]  Real={counters['Real']}  "
                  f"Fake={counters['Fake']}  AI={counters['AI_Generated']}")

        # Stop once all classes are full
        if all(v >= MAX_PER_CLASS for v in counters.values()):
            print("  [INFO] All classes reached cap. Stopping.")
            break

    except Exception as e:
        print(f"  [WARN] sample {i} skipped: {e}")

print("\n" + "=" * 60)
print(f"  Done!  Real={counters['Real']}  Fake={counters['Fake']}  AI_Generated={counters['AI_Generated']}")
print(f"  Saved in: {BASE_DIR}")
print("=" * 60)

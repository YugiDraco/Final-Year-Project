# 🔬 Advanced Multi-Modal AI Forensic System v4.1

Unified Media Authenticity Engine for distinguishing **REAL**, **DEEPFAKE** (Face-Swap), and **AI-GENERATED** (Synthetic) images.

## 🌟 Key Features
- **3-Class Classification**: Identifies Real, Deepfake (manipulated), and AI-Generated (fully synthetic/GAN/Diffusion) media.
- **Probabilistic Dominance Rule**: Uses a sophisticated fusion engine to weigh CNN probabilities against forensic spectral/noise signals.
- **Forensic Dossier**: Provides explainable heatmaps including **Grad-CAM (Attention)**, **FFT (Frequency)**, and **Noise Residuals**.
- **Automated Grading**: Generates a forensic grade (**S** to **C**) with natural language reasoning.
- **PDF/JSON Reporting**: Archives every analysis with detailed forensic metrics.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Prj_demo
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Requirement

The system is optimized for the **OpenFake** dataset. Before training or advanced testing, you should download and organize the data.

### 1. Automatic Download
Run the included script to download and organize images from HuggingFace (`ComplexDataLab/OpenFake`):
```bash
python download_dataset.py
```
This script will create the following directory structure in your `d:/Final Year Project/DeepFake Analysis/Dataset` path (configurable in `config.py`):
```text
Dataset/
├── Real/           (Original photographs)
├── Fake/           (Face-swapped / Reactment)
└── AI_Generated/   (Fully synthetic / GAN / Diffusion)
```

### 2. Manual Setup
If downloading manually, ensure the labels are distinguished as follows:
- **Deepfake**: Specific facial manipulations (e.g., SIMSWAP, InsightFace).
- **AI-Gen**: Text-to-image or fully synthetic generations (e.g., Midjourney, DALL-E, StyleGAN).

## 🚀 Usage

### Launch Gradio Web UI
The primary interface for analysis:
```bash
python app.py
```
Open `http://localhost:7861` in your browser.

### Run Verification Suite
To verify the 3-class identification logic:
```bash
python test_classification.py
```

### Batch Analysis
Analyze a folder of images and generate JSON reports:
```bash
python main.py --input path/to/folder
```

## 📁 Code Structure
- `app.py`: Gradio web interface and analysis pipeline.
- `fusion.py`: **The Brain.** Decision fusion engine using the Dominance Rule.
- `forensics.py`: Extracts FFT, ELA, and Noise signals.
- `models.py`: EfficientNet-B0 implementation and inference logic.
- `grading.py`: Smart grading system (S/A/B/C).
- `explainability.py`: Generates heatmaps (Grad-CAM, FFT, Noise).
- `config.py`: Centralized paths, thresholds, and API keys.

---
**Engine v4.1** | *Designed for high-precision media forensic analysis.*

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, datasets
from torchvision.models import inception_v3, Inception_V3_Weights
import os

# =========================
# PARAMETERS
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "papaya main dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "papaya_inceptionv3_pytorch.pth")
print(f"DEBUG: MODEL_PATH={MODEL_PATH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DYNAMIC CLASS NAMES
# =========================
CLASS_NAMES = sorted(os.listdir(DATA_DIR))  # Automatically read folder names
NUM_CLASSES = len(CLASS_NAMES)

# Example recommendations; you can expand this dict for each class
default_recommendation = "No recommendation available for this class."
recommendations = {
    'healthy': 'No action needed. Keep monitoring the plant.',
    'leaf_spot': 'Apply fungicide and remove infected leaves.',
    'mosaic_virus': 'Remove infected plants to prevent spread.',
    'powdery_mildew': 'Spray neem oil or suitable fungicide.',
    'Yellow_Necrotic_Spots_Holes': 'Remove affected leaves and apply treatment.'
}

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# LOAD MODEL
# =========================
weights = Inception_V3_Weights.IMAGENET1K_V1
model = inception_v3(weights=weights, aux_logits=True)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image: Image.Image):
    img = image.convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        if isinstance(outputs, tuple):  # aux_logits=True returns tuple
            outputs = outputs[0]
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Papaya Leaf Disease Detection", layout="centered")
st.title("🍃 Papaya Leaf Disease Detection")
st.write("Upload a papaya leaf image and get prediction!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)

    st.write("Predicting...")
    prediction = predict_image(image)

    st.success(f"Predicted class: **{prediction}**")

    # Show recommendation (use fallback if class not in dict)
    st.info(f"Recommendation: {recommendations.get(prediction, default_recommendation)}")
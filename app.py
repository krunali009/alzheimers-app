import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
import io

st.title("Alzheimer's Detection App (SVM + HOG)")

# Load model and metadata
@st.cache_resource
def load_model():
    model = joblib.load("alzheimer_svm_model.joblib")
    le = joblib.load("label_encoder.joblib")
    classes = list(np.load("class_names.npy", allow_pickle=True))
    return model, le, classes

model, le, classes = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image", type=["png","jpg","jpeg"])

def preprocess(pil_img, size=(128,128)):
    img_gray = pil_img.convert("L").resize(size)
    return np.array(img_gray)

def extract_features(img_array):
    return hog(img_array, orientations=8, pixels_per_cell=(16,16),
               cells_per_block=(1,1), block_norm='L2-Hys')

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    arr = preprocess(image)
    feat = extract_features(arr).reshape(1, -1)

    pred = model.predict(feat)[0]
    pred_label = le.inverse_transform([pred])[0]
    st.success(f"Prediction: {pred_label}")

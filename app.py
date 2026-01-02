import streamlit as st
import pickle
import numpy as np
from PIL import Image
import base64

# Load model
with open("logi_reg.pkl", "rb") as f:
    model = pickle.load(f)
    # ---------- BACKGROUND IMAGE ----------
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Title highlight */
        .main-title {{
            background: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 42px;
            font-weight: 800;
            text-shadow: 0px 0px 12px #00eaff;
        }}

        /* Output visibility */
        .stAlert {{
            background-color: rgba(0, 0, 0, 0.65) !important;
            color: white !important;
            border-radius: 12px;
        }}

        /* File uploader */
        section[data-testid="stFileUploader"] {{
            background: rgba(0,0,0,0.55);
            padding: 15px;
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("background.png")   # your generated image (NO watermark)

st.markdown(
    '<div class="main-title">🦴 Osteoporosis Detection App</div>',
    unsafe_allow_html=True
)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((300, 300))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, -1)
    return img_array

def is_valid_xray(image):
    """
    Soft validation:
    Accept grayscale-dominant medical images
    """
    img = np.array(image.convert("RGB")).astype("float32")

    # Difference between RGB channels
    diff_rg = np.mean(np.abs(img[:, :, 0] - img[:, :, 1]))
    diff_rb = np.mean(np.abs(img[:, :, 0] - img[:, :, 2]))
    diff_gb = np.mean(np.abs(img[:, :, 1] - img[:, :, 2]))

    avg_diff = (diff_rg + diff_rb + diff_gb) / 3

    # X-rays have very low channel difference
    return avg_diff < 15


uploaded_file = st.file_uploader(
    "Upload Bone X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ❌ INVALID IMAGE CHECK
        if not is_valid_xray(image):
            st.error("❌ Please upload a valid bone X-ray image")
        else:
            img = preprocess_image(image)
            pred = model.predict(img)

            st.success(f"Prediction: **{pred[0]}**")

    except Exception:
        st.error("❌ Please upload a valid bone X-ray image")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Pill Counter App", layout="wide")

# Load Model
model = YOLO("best.pt")

# Custom CSS for Modern UI
st.markdown("""
<style>
    .main {background: #0d1117; color: white;}
    .title {text-align: center; font-size: 38px; font-weight: bold; color: #4CAF50;}
    .counter-box {
        padding: 15px;
        background: #1e1e1e;
        border-radius: 12px;
        margin-top: 10px;
        font-size: 20px;
        color: white;
        text-align: center;
        border: 2px solid #4CAF50;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 Smart Pill & Capsule Detection</p>', unsafe_allow_html=True)

option = st.sidebar.selectbox("Select Mode", ["Image Upload", "Real-time Camera"])

# Helper Function to Count Classes
def get_counts(results):
    pills = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0:
            pills += 1
        elif cls == 1:
            capsules += 1
    return pills, capsules, pills + capsules

# IMAGE UPLOAD MODE ----------------------------------------------------------
if option == "Image Upload":
    st.subheader("📸 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        results = model(img_np)
        annotated = results[0].plot()

        pills, capsules, total = get_counts(results)

        st.image(annotated, caption="Detection Results", use_column_width=True)

        st.markdown(f"""
        <div class="counter-box">
            Tablets: {pills} <br>
            Capsules: {capsules} <br>
            <b>Total: {total}</b>
        </div>
        """, unsafe_allow_html=True)

# REAL-TIME CAMERA MODE ------------------------------------------------------
else:
    st.subheader("🎥 Real-time Detection")

    camera = st.checkbox("Start Camera")

    if camera:
        frame_window = st.empty()

        cap = cv2.VideoCapture(0)

        while camera:
            ret, frame = cap.read()
            if not ret:
                st.write("Camera not detected!")
                break

            results = model(frame)
            pills, capsules, total = get_counts(results)

            annotated = results[0].plot()

            cv2.putText(annotated, f"Tablets: {pills} | Capsules: {capsules} | Total: {total}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            frame_window.image(annotated, channels="BGR")

        cap.release()

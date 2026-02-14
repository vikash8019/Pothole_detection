import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pothole Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* FONT */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN BACKGROUND - SOFT GREY */
.stApp {
    background-color: #f3f4f6;
    color: #111827;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #e5e7eb;
    border-right: 1px solid #d1d5db;
}

/* HEADINGS */
h1, h2, h3 {
    color: #111827;
    font-weight: 600;
}

/* BUTTON */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 2.8em;
    width: 12em;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transition: 0.2s;
}

/* METRIC CARDS */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* TABS */
button[role="tab"] {
    background: #e5e7eb !important;
    color: #374151 !important;
    border-radius: 6px;
    margin: 2px;
}
button[aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

/* SLIDER */
[data-baseweb="slider"] {
    color: #2563eb;
}

/* FOOTER HIDE */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)




st.markdown("# 🚧 Pothole Detection Dashboard")

# ---------------- LOAD MODEL ----------------
model = YOLO("model/best.pt")   # PATH HERE

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")

# -------- AUTO / MANUAL CONFIDENCE --------
mode = st.sidebar.radio("Detection Mode", ["Auto", "Manual"])

if mode == "Auto":
    confidence = 0.25
    st.sidebar.write("Auto Confidence: 0.25")
else:
    confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

box_color = st.sidebar.color_picker("Box Color", "#FF0000")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🖼 Image", "🎥 Video", "📷 Live Camera"])

# ================= IMAGE TAB =================
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image")

        if st.button("Detect Potholes"):
            with st.spinner("Detecting..."):
                results = model(image, conf=confidence, max_det=50, iou=0.7)
                annotated = results[0].plot()

                count = len(results[0].boxes)

            with col2:
                st.image(annotated, caption="Detected Image")

            st.metric("Potholes Detected", count)

            # Download
            result_img = Image.fromarray(annotated)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                "Download Result",
                byte_im,
                file_name="pothole_result.png"
            )

# ================= VIDEO TAB =================
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, max_det=50, iou=0.7)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR")

        cap.release()

# ================= LIVE CAMERA TAB =================
with tab3:
    run = st.checkbox("Start Camera")
    frame_window = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        results = model(frame, conf=confidence, max_det=50, iou=0.7)
        annotated = results[0].plot()

        frame_window.image(annotated, channels="BGR")

    camera.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ | Pothole Detection AI")

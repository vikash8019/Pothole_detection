import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pothole Detection AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚧"
)

# ---------------- ENHANCED CUSTOM CSS ----------------
st.markdown("""
<style>

/* MODERN FONTS */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND - GRADIENT */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    color: #1a202c;
}

/* MAIN CONTENT CONTAINER */
.main .block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem 3rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* SIDEBAR - GLASS MORPHISM */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* HEADINGS WITH BETTER VISIBILITY */
h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 3rem !important;
    color: #1a202c !important;
    margin-bottom: 0.5rem !important;
    text-align: center;
    animation: fadeInDown 0.8s ease;
    text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);
}

h2 {
    font-family: 'Space Grotesk', sans-serif;
    color: #1a202c !important;
    font-weight: 600;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
}

h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #2d3748 !important;
    font-weight: 600;
}

h4 {
    font-family: 'Space Grotesk', sans-serif;
    color: #2d3748 !important;
    font-weight: 600;
}

/* ANIMATED SUBTITLE */
.subtitle {
    text-align: center;
    color: #4a5568;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    animation: fadeIn 1s ease;
    font-weight: 500;
}

/* ENHANCED BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.stButton>button:active {
    transform: translateY(0);
}

/* DOWNLOAD BUTTON */
.stDownloadButton>button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    transition: all 0.3s ease;
}

.stDownloadButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
}

/* METRIC CARDS - ENHANCED */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
    border: 2px solid #e2e8f0;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
}

div[data-testid="metric-container"] label {
    font-size: 0.9rem !important;
    color: #4a5568 !important;
    font-weight: 600 !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #1a202c !important;
}

/* TABS - MODERN LOOK */
button[role="tab"] {
    background: #e2e8f0 !important;
    color: #4a5568 !important;
    border-radius: 10px !important;
    margin: 4px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

button[role="tab"]:hover {
    background: #cbd5e0 !important;
    transform: translateY(-2px);
}

button[aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

/* FILE UPLOADER - ENHANCED */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #f7fafc 0%, #ffffff 100%);
    padding: 24px;
    border-radius: 16px;
    border: 2px dashed #cbd5e0;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #667eea;
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
}

/* SLIDER - CUSTOM STYLE */
[data-baseweb="slider"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stSlider > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* RADIO BUTTONS */
.stRadio > label {
    font-weight: 600;
    color: white !important;
}

/* COLOR PICKER */
.stColorPicker > label {
    font-weight: 600;
    color: white !important;
}

/* IMAGE CONTAINERS */
.stImage {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease;
}

.stImage:hover {
    transform: scale(1.02);
}

/* SPINNER */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* CAMERA INPUT */
[data-testid="stCameraInput"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

/* ANIMATIONS */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

/* ALERT/INFO BOXES */
.stAlert {
    border-radius: 12px;
    border-left: 4px solid #667eea;
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #2d3748;
    font-size: 0.9rem;
    animation: fadeIn 1.5s ease;
}

.footer h4 {
    color: #1a202c !important;
    margin-bottom: 0.5rem;
}

.footer p {
    color: #4a5568 !important;
}

/* HIDE STREAMLIT BRANDING */
footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* INFO CARD */
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    animation: fadeInDown 0.8s ease;
}

.info-card h3 {
    color: white !important;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER WITH ANIMATION ----------------
st.markdown("# 🚧 Pothole Detection AI")
st.markdown('<p class="subtitle">Intelligent Road Infrastructure Analysis System | Powered by YOLOv8</p>', unsafe_allow_html=True)

# ---------------- INFO CARD ----------------
st.markdown("""
<div class="info-card">
    <h3>🎯 About This System</h3>
    <p>Advanced AI-powered pothole detection system using state-of-the-art computer vision technology. 
    Upload images, videos, or use live camera to detect road damage in real-time.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- SIDEBAR WITH ENHANCED STYLING ----------------
st.sidebar.markdown("## ⚙️ Detection Settings")
st.sidebar.markdown("---")

# -------- AUTO / MANUAL CONFIDENCE --------
mode = st.sidebar.radio("🎚️ Detection Mode", ["Auto", "Manual"], help="Choose automatic or manual confidence threshold")

if mode == "Auto":
    confidence = 0.25
    st.sidebar.info("📊 Auto Confidence: 0.25")
else:
    confidence = st.sidebar.slider("🔍 Confidence Threshold", 0.1, 1.0, 0.4, 0.05,
                                   help="Adjust detection sensitivity")

st.sidebar.markdown("---")
box_color = st.sidebar.color_picker("🎨 Detection Box Color", "#FF0000")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
**Model:** YOLOv8  
**Task:** Object Detection  
**Classes:** Pothole  
**Input:** Image/Video
""")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "🎥 Video Analysis", "📷 Live Camera"])

# ================= IMAGE TAB =================
with tab1:
    st.markdown("### Upload an image to detect potholes")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG"
        )
    
    with col_info:
        st.info("💡 **Tips:**\n- Use clear images\n- Good lighting helps\n- Close-up shots work best")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Original Image")
            st.image(image, use_container_width=True)
        
        st.markdown("")
        
        if st.button("🔍 Detect Potholes", use_container_width=True):
            with st.spinner("🔄 Analyzing image... Please wait"):
                results = model(image, conf=confidence, max_det=50, iou=0.7)
                annotated = results[0].plot()
                count = len(results[0].boxes)
            
            with col2:
                st.markdown("#### ✅ Detection Results")
                st.image(annotated, use_container_width=True)
            
            # Metrics
            st.markdown("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("🕳️ Potholes Detected", count)
            
            with col_m2:
                st.metric("📊 Confidence Level", f"{confidence:.2f}")
            
            with col_m3:
                status = "⚠️ Action Needed" if count > 0 else "✅ Road Clear"
                st.metric("🚦 Status", status)
            
            # Download
            st.markdown("---")
            result_img = Image.fromarray(annotated)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                "💾 Download Detection Result",
                byte_im,
                file_name="pothole_detection_result.png",
                mime="image/png",
                use_container_width=True
            )

# ================= VIDEO TAB =================
with tab2:
    st.markdown("### Upload a video for frame-by-frame analysis")
    
    col_upload_v, col_info_v = st.columns([2, 1])
    
    with col_upload_v:
        uploaded_video = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "mov", "avi"],
            help="Supported formats: MP4, MOV, AVI"
        )
    
    with col_info_v:
        st.warning("⚠️ **Note:**\n- Large videos may take time\n- Processing is done frame-by-frame")
    
    if uploaded_video:
        st.success("✅ Video uploaded successfully!")
        
        if st.button("🎬 Start Video Analysis", use_container_width=True):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            st.markdown("#### 🎥 Processing Video...")
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            total_detections = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, conf=confidence, max_det=50, iou=0.7)
                annotated = results[0].plot()
                total_detections += len(results[0].boxes)
                
                stframe.image(annotated, channels="BGR", use_container_width=True)
                
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
            
            cap.release()
            
            st.success("✅ Video analysis complete!")
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("🎞️ Total Frames", total_frames)
            with col_v2:
                st.metric("🕳️ Total Detections", total_detections)

# ================= LIVE CAMERA TAB =================
with tab3:
    st.markdown("### Use your camera for real-time detection")
    
    st.info("📸 Click the button below to capture an image from your camera")
    
    img_file = st.camera_input("Take a Picture")
    
    if img_file is not None:
        image = Image.open(img_file)
        
        with st.spinner("🔄 Analyzing captured image..."):
            results = model(image, conf=confidence, max_det=50, iou=0.7)
            annotated = results[0].plot()
            pothole_count = len(results[0].boxes)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Captured Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### ✅ Detection Results")
            st.image(annotated, use_container_width=True)
        
        # Metrics
        st.markdown("---")
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.metric("🕳️ Potholes Detected", pothole_count)
        
        with col_c2:
            st.metric("📊 Confidence Level", f"{confidence:.2f}")
        
        with col_c3:
            status = "⚠️ Report Needed" if pothole_count > 0 else "✅ Road OK"
            st.metric("🚦 Status", status)
        
        # Download Result
        st.markdown("---")
        result_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            "💾 Download Camera Result",
            byte_im,
            file_name="camera_pothole_result.png",
            mime="image/png",
            use_container_width=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <h4>🚧 Pothole Detection AI System</h4>
    <p>Made with ❤️ | Powered by YOLOv8 & Streamlit</p>
    <p style="font-size: 0.8rem; color: #a0aec0;">Helping make roads safer, one detection at a time</p>
</div>
""", unsafe_allow_html=True)


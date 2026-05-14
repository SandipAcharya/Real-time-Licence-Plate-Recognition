import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os

from detection import LicensePlateDetector
from ocr_easyocr import EasyOCRReader
from ocr_legacy import LegacySVMReader
from database import PlateDatabase

st.set_page_config(page_title="Nepali ALPR System", layout="wide")

# --- Initialization ---
@st.cache_resource
def load_models():
    detector = LicensePlateDetector(model_path='models/yolov8n.pt') # Mock model for now, replace with fine-tuned
    ocr_modern = EasyOCRReader(languages=['ne', 'en'])
    ocr_legacy = LegacySVMReader(model_path='models/svm_model.pkl')
    return detector, ocr_modern, ocr_legacy

detector, ocr_modern, ocr_legacy = load_models()
db = PlateDatabase()

# --- Helper Functions ---
def process_image(image, use_legacy=False):
    # Detect plates
    st.write("Detecting plates...")
    detections, cropped_plates = detector.detect(image)
    
    img_with_boxes = detector.draw_detections(image, detections)
    
    results = []
    
    # OCR on each detected plate
    for i, cropped in enumerate(cropped_plates):
        st.write(f"Processing plate {i+1}...")
        
        # Modern OCR
        t0 = time.time()
        text_mod, conf_mod = ocr_modern.read_text(cropped)
        t_mod = time.time() - t0
        
        # Legacy OCR
        text_leg = ""
        conf_leg = 0.0
        t_leg = 0.0
        if use_legacy:
            t0 = time.time()
            text_leg, conf_leg = ocr_legacy.read_text(cropped)
            t_leg = time.time() - t0
            
        results.append({
            'cropped': cropped,
            'text_modern': text_mod,
            'conf_modern': conf_mod,
            'time_modern': t_mod,
            'text_legacy': text_leg,
            'conf_legacy': conf_leg,
            'time_legacy': t_leg
        })
        
        # Log to database (we log the modern one as default, but you can choose)
        db.log_detection(text_mod, conf_mod, cropped, ocr_method="EasyOCR")
        
    return img_with_boxes, results


# --- UI Layout ---
st.title("🚘 Real-Time Nepali Number Plate Recognition")
st.markdown("Welcome to the automated ALPR system. Upload an image to detect and register Nepali number plates.")

tab1, tab2 = st.tabs(["🔍 Detection & OCR", "🗄️ Database Records"])

with tab1:
    st.sidebar.header("Settings")
    compare_mode = st.sidebar.checkbox("Compare with Legacy (HOG+SVM)", value=True, 
                                       help="Run the old contour-based SVM alongside the new EasyOCR deep learning model for comparison.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if st.button("Run ALPR", type="primary"):
                with st.spinner("Processing..."):
                    img_result, ocr_results = process_image(image, use_legacy=compare_mode)
                    
                st.subheader("Detection Results")
                st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Extracted Data")
                    for i, res in enumerate(ocr_results):
                        st.markdown(f"**Plate #{i+1}**")
                        st.image(cv2.cvtColor(res['cropped'], cv2.COLOR_BGR2RGB), width=200)
                        
                        if compare_mode:
                            st.success(f"**Modern (EasyOCR):** {res['text_modern']} (Conf: {res['conf_modern']:.2f}) - {res['time_modern']:.2f}s")
                            st.warning(f"**Legacy (HOG+SVM):** {res['text_legacy']} (Conf: {res['conf_legacy']:.2f}) - {res['time_legacy']:.2f}s")
                        else:
                            st.success(f"**Plate Text:** {res['text_modern']} (Conf: {res['conf_modern']:.2f})")
                        st.divider()
                        
with tab2:
    st.header("Automated Registration Database")
    st.markdown("This database automatically logs all recognized license plates without manual entry.")
    
    if st.button("Refresh Database"):
        pass # Streamlit reruns on button click anyway
        
    records = db.get_all_logs()
    
    if records:
        # Convert to pandas DataFrame for nice display
        df = pd.DataFrame(records, columns=['ID', 'Plate Text', 'Confidence', 'Timestamp', 'Image Path', 'OCR Method'])
        # We don't need to show image path in the table
        st.dataframe(df[['ID', 'Plate Text', 'Confidence', 'OCR Method', 'Timestamp']], use_container_width=True)
        
        # Gallery of recent detections
        st.subheader("Recent Detections")
        cols = st.columns(4)
        for i, row in enumerate(records[:8]): # show last 8
            if row[4] and os.path.exists(row[4]):
                img = Image.open(row[4])
                cols[i%4].image(img, caption=f"{row[1]} ({row[3]})", use_container_width=True)
    else:
        st.info("Database is currently empty. Run a detection to populate it.")

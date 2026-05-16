import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os

from detection import LicensePlateDetector
from ocr_easyocr import EasyOCRReader
from ocr_legacy import LegacySVMReader, RobustSVMReader
from ocr_tesseract import TesseractReader
from database import PlateDatabase
from database import PlateDatabase

st.set_page_config(page_title="Nepali ALPR System", layout="wide")

# --- Initialization ---
@st.cache_resource
def load_models():
    detector = LicensePlateDetector(model_path='models/best.pt')
    ocr_modern = EasyOCRReader(languages=['ne', 'en'])
    ocr_legacy = LegacySVMReader(model_path='models/svm_model.pkl')
    ocr_robust = RobustSVMReader(model_path='models/svm_model.pkl')
    ocr_tesseract = TesseractReader(language='nep')
    return detector, ocr_modern, ocr_legacy, ocr_robust, ocr_tesseract

detector, ocr_modern, ocr_legacy, ocr_robust, ocr_tesseract = load_models()
db = PlateDatabase()

# --- Helper Functions ---
def pad_image(image, padding=20):
    """Add white padding around image for better OCR results"""
    if len(image.shape) == 3:
        return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255])

def enhance_image(image):
    """Upscale and sharpen blurry license plate crops for better deep learning OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upscale by 2x
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def process_image(image, use_legacy=False, is_cropped=False):
    results = []
    
    if is_cropped:
        # Bypass YOLO, use the whole image as the cropped plate
        st.write("Processing pre-cropped plate...")
        detections = []
        cropped_plates = [image]
        img_with_boxes = image.copy()
    else:
        # Detect plates using YOLO
        st.write("Detecting plates...")
        detections, cropped_plates = detector.detect(image)
        img_with_boxes = detector.draw_detections(image, detections)
    
    results = []
    
    # OCR on each detected plate
    for i, cropped in enumerate(cropped_plates):
        st.write(f"Processing plate {i+1}...")
        
        # Enhance and pad for Modern OCRs
        enhanced_cropped = enhance_image(cropped)
        padded_cropped = pad_image(enhanced_cropped)
        
        # Modern OCR (EasyOCR)
        t0 = time.time()
        text_mod, conf_mod = ocr_modern.read_text(padded_cropped)
        t_mod = time.time() - t0
        
        # Tesseract OCR
        t0 = time.time()
        text_tess, conf_tess = ocr_tesseract.read_text(padded_cropped)
        t_tess = time.time() - t0

        # Legacy OCR (Contour + HOG + SVM)
        t0 = time.time()
        text_leg, conf_leg, chars_leg = ocr_legacy.read_text(cropped)
        t_leg = time.time() - t0
            
        # Robust OCR (Projection + HOG + SVM)
        t0 = time.time()
        text_rob, conf_rob, chars_rob = ocr_robust.read_text(cropped)
        t_rob = time.time() - t0
            
        results.append({
            'cropped': cropped,
            'text_modern': text_mod,
            'conf_modern': conf_mod,
            'time_modern': t_mod,
            'text_tesseract': text_tess,
            'conf_tesseract': conf_tess,
            'time_tesseract': t_tess,
            'text_legacy': text_leg,
            'conf_legacy': conf_leg,
            'time_legacy': t_leg,
            'chars_legacy': chars_leg,
            'text_robust': text_rob,
            'conf_robust': conf_rob,
            'time_robust': t_rob,
            'chars_robust': chars_rob
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
    compare_mode = st.sidebar.checkbox("Show Multi-Method Comparison", value=True, 
                                       help="Compare Legacy SVM, Robust SVM, EasyOCR, and Tesseract side-by-side.")
    
    st.sidebar.divider()
    image_type = st.sidebar.radio("Input Image Type", 
                                  ["Full Vehicle Image", "Already Cropped Plate"],
                                  help="Select 'Already Cropped Plate' if you are uploading an image that is just the license plate. This bypasses the YOLO detection phase.")
    
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
                    is_cropped = (image_type == "Already Cropped Plate")
                    img_result, ocr_results = process_image(image, use_legacy=compare_mode, is_cropped=is_cropped)
                    
                st.subheader("Detection Results")
                st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Extracted Data")
                    for i, res in enumerate(ocr_results):
                        st.markdown(f"**Plate #{i+1}**")
                        st.image(cv2.cvtColor(res['cropped'], cv2.COLOR_BGR2RGB), width=200)
                        
                        if compare_mode:
                            st.write("---")
                            st.markdown("##### ✂️ Segmentation Comparison")
                            col_c, col_p = st.columns(2)
                            with col_c:
                                st.caption("Old Contour Method:")
                                if res.get('chars_legacy'):
                                    st.image(res['chars_legacy'], width=30)
                                else:
                                    st.caption("*(No characters extracted)*")
                            with col_p:
                                st.caption("New Projection Method:")
                                if res.get('chars_robust'):
                                    st.image(res['chars_robust'], width=30)
                                else:
                                    st.caption("*(No characters extracted)*")
                            
                            st.write("---")
                            st.success(f"**EasyOCR:** {res['text_modern']} (Conf: {res['conf_modern']:.2f}) - {res['time_modern']:.2f}s")
                            st.info(f"**Tesseract OCR:** {res['text_tesseract']} (Conf: {res['conf_tesseract']:.2f}) - {res['time_tesseract']:.2f}s")
                            st.warning(f"**Legacy (Contour+SVM):** {res['text_legacy']} (Conf: {res['conf_legacy']:.2f}) - {res['time_legacy']:.2f}s")
                            st.error(f"**Robust (Proj+SVM):** {res['text_robust']} (Conf: {res['conf_robust']:.2f}) - {res['time_robust']:.2f}s")
                        else:
                            st.success(f"**Plate Text (Robust):** {res['text_robust']} (Conf: {res['conf_robust']:.2f})")
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

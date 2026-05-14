# Real-Time Nepali Number Plate Recognition

## Overview

This project focuses on developing a **Real-Time Nepali Number Plate Recognition** system. It leverages advanced techniques in object detection and optical character recognition (OCR) to identify and recognize Nepali number plates written in the Devanagari script.

The system has been recently upgraded from a legacy YOLOv7 + HOG/SVM pipeline to a modern, robust architecture utilizing **YOLOv8** for detection, **EasyOCR** for end-to-end Devanagari text extraction, and an **Automated SQLite Database** for seamless registration logging.

## System Architecture
   ![System Architecture](datasets_link/image.png)

### 1. License Plate Detection (YOLOv8)
- **Description**: We utilize Ultralytics YOLOv8, a state-of-the-art object detection model, to detect and crop the license plate from the input image or video frame.
- **Justification**: YOLOv8 offers superior speed and accuracy compared to older models like YOLOv7, making it perfect for real-time edge deployment.

### 2. Optical Character Recognition (EasyOCR vs Legacy SVM)
- **Modern Approach (EasyOCR)**: An end-to-end deep learning approach that natively supports Devanagari/Nepali script. It is highly robust to different fonts, lighting conditions, and skew, entirely removing the need for manual contour segmentation.
- **Legacy Approach (HOG + SVM)**: For comparison purposes, the older method utilizing contour-based character segmentation and a Support Vector Machine trained on Histogram of Oriented Gradients (HOG) features is still available in the codebase (`src/ocr_legacy.py`).

### 3. Automated Database Logging
- **Description**: Detected plates are automatically logged into a local SQLite database (`logs/plates.db`).
- **Features**: Logs the recognized text, confidence score, timestamp, and saves a cropped snapshot of the plate. This eliminates the need for manual data entry at toll booths or security gates.

## Project Structure

```
Real-time-Licence-Plate-Recognition/
├── models/                # Saved model weights (yolov8.pt, svm_model.pkl)
├── src/                   # Core application code
│   ├── app.py             # Streamlit Web UI
│   ├── detection.py       # YOLOv8 inference wrapper
│   ├── ocr_easyocr.py     # Modern EasyOCR pipeline
│   ├── ocr_legacy.py      # Legacy contour+SVM pipeline
│   └── database.py        # SQLite automated logger
├── logs/                  # Database file and cropped snapshots
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

*(Note: The original Jupyter notebooks `project.ipynb` and `ocr.ipynb` are retained in the root directory for historical reference).*

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Real-time-Licence-Plate-Recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Launch the interactive Streamlit dashboard:
   ```bash
   streamlit run src/app.py
   ```

## Web Interface

The Streamlit app features two main tabs:
1. **Detection & OCR**: Upload an image to run the pipeline. You can toggle "Compare with Legacy" to see a side-by-side performance comparison between EasyOCR and the HOG+SVM method.
2. **Database Records**: A live view of the automated SQLite database displaying all historically detected plates and their cropped image snapshots.

## Future Enhancements
- Fine-tune YOLOv8 specifically on custom Nepali fonts.
- Add real-time webcam/RTSP video stream processing directly in the Streamlit app.
- Export database logs to CSV or an external cloud database.

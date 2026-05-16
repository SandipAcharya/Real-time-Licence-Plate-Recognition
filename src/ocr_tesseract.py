import pytesseract
from PIL import Image
import cv2
import sys

# Configure tesseract path for Windows
if sys.platform.startswith('win'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TesseractReader:
    def __init__(self, language='nep'):
        self.language = language

    def read_text(self, image):
        """
        Reads Nepali text from an image using Tesseract OCR.
        Returns (text, confidence)
        """
        try:
            # Tesseract expects PIL Image or ndarray
            # For best results, use PSM 6 (Assume a single uniform block of text)
            custom_config = r'--oem 3 --psm 6'
            
            # Get verbose data to calculate confidence
            data = pytesseract.image_to_data(image, lang=self.language, config=custom_config, output_type=pytesseract.Output.DICT)
            
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > -1: # Ignore empty/invalid results
                    text_parts.append(data['text'][i])
                    confidences.append(float(data['conf'][i]))
            
            final_text = " ".join(text_parts).strip()
            
            # Calculate average confidence
            avg_conf = (sum(confidences) / len(confidences)) / 100.0 if confidences else 0.0
            
            return final_text, avg_conf
            
        except pytesseract.TesseractNotFoundError:
            return "Error: Tesseract not installed. Please install Tesseract OCR.", 0.0
        except Exception as e:
            return f"Tesseract Error: {str(e)}", 0.0

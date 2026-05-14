import easyocr
import cv2

class EasyOCRReader:
    def __init__(self, languages=['ne', 'en']):
        """
        Initialize EasyOCR reader with Nepali and English language support.
        """
        self.reader = easyocr.Reader(languages, gpu=True) # Set gpu=False if no CUDA

    def read_text(self, image):
        """
        Read text from a cropped license plate image.
        
        Args:
            image (numpy.ndarray): Cropped BGR image of the license plate.
            
        Returns:
            text (str): The concatenated recognized text.
            confidence (float): Average confidence of the reading.
        """
        # EasyOCR prefers RGB or Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # You could add more preprocessing here if needed (e.g., thresholding)
        
        results = self.reader.readtext(gray)
        
        if not results:
            return "", 0.0
            
        text = " ".join([res[1] for res in results])
        conf = sum([res[2] for res in results]) / len(results)
        
        return text, conf

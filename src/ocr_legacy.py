import cv2
import numpy as np
import os
import pickle

class LegacySVMReader:
    def __init__(self, model_path='models/svm_model.pkl'):
        """
        Initialize the legacy HOG + SVM reader.
        """
        self.model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print(f"[Warning] Legacy SVM model not found at {model_path}. Returning mocked results.")
            
        # Defining standard dimension for HOG
        self.image_size = (28, 28)

    def extract_hog_features(self, image):
        """
        Extract HOG features from a character image.
        """
        # Note: These parameters should match exactly what was used during training
        hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(14, 14),
            _blockStride=(7, 7),
            _cellSize=(7, 7),
            _nbins=9
        )
        return hog.compute(image).flatten()

    def segment_characters(self, image):
        """
        Contour-based segmentation to extract individual characters.
        """
        img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.bitwise_not(img_binary_lp)

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]
        
        # Dimensions estimation for characters
        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]

        contours, _ = cv2.findContours(img_binary_lp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        char_images = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Rough filter for character size
            if (dimensions[0] < h < dimensions[1]) and (dimensions[2] < w < dimensions[3]):
                char_img = img_binary_lp[y:y+h, x:x+w]
                char_img = cv2.resize(char_img, self.image_size)
                char_images.append(char_img)
                
        return char_images

    def read_text(self, image):
        """
        Read text from a cropped license plate using HOG+SVM.
        
        Args:
            image (numpy.ndarray): Cropped BGR image.
            
        Returns:
            text (str): The recognized text.
            confidence (float): Mocked/average confidence.
        """
        if self.model is None:
            # Fallback mock for demonstration if model is missing
            return "SVM_MISSING", 0.0
            
        char_images = self.segment_characters(image)
        if not char_images:
            return "", 0.0
            
        predictions = []
        for char_img in char_images:
            features = self.extract_hog_features(char_img)
            # SVM predict
            pred = self.model.predict([features])
            predictions.append(str(pred[0]))
            
        text = "".join(predictions)
        return text, 0.8 # Legacy didn't easily return probabilities depending on SVM config

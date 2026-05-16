import cv2
import numpy as np
import os
import joblib
from skimage.feature import hog
from segmentation import ProjectionSegmenter

class LegacySVMReader:
    def __init__(self, model_path='models/svm_model.pkl'):
        """
        Initialize the legacy HOG + SVM reader.
        """
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"[Warning] Failed to load legacy SVM model at {model_path}: {e}. Returning mocked results.")
                self.model = None
        else:
            print(f"[Warning] Legacy SVM model not found at {model_path}. Returning mocked results.")

    def extract_hog_features(self, image):
        """
        Extract HOG features from a character image using exact notebook logic.
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        # The notebook resized to 64x128
        img_res = cv2.resize(img_gray, (64, 128), interpolation=cv2.INTER_AREA)
        
        # Use skimage hog to match training
        hog_img = hog(img_res, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1, 1))
        return hog_img

    def _normalize_binary(self, binary):
        """Ensure the background is black and text is white."""
        top = binary[0:3, :]
        bottom = binary[-3:, :]
        left = binary[:, 0:3]
        right = binary[:, -3:]
        border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
        
        if np.mean(border_pixels) > 127:
            # Background is white, invert to make background black
            return cv2.bitwise_not(binary)
        return binary

    def segment_characters(self, image):
        """
        Contour-based segmentation to extract individual characters.
        Matches the logic found in the original notebook.
        """
        img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur slightly to reduce noise
        blur = cv2.GaussianBlur(img_gray_lp, (3,3), 0)
        _, img_binary_lp = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize polarity: Ensure black background, white text
        img_binary_lp = self._normalize_binary(img_binary_lp)
        
        # Ensure kernel is a proper numpy array
        kernel = np.ones((3, 3), np.uint8)
        img_binary_lp = cv2.erode(img_binary_lp, kernel)
        img_binary_lp = cv2.dilate(img_binary_lp, kernel)

        LP_WIDTH = img_binary_lp.shape[0] # Height
        LP_HEIGHT = img_binary_lp.shape[1] # Width
        
        # Make borders black to remove boundary noise (assuming white text on black bg)
        img_binary_lp[0:3, :] = 0
        img_binary_lp[:, 0:3] = 0
        img_binary_lp[LP_WIDTH - 3:LP_WIDTH, :] = 0
        img_binary_lp[:, LP_HEIGHT - 3:LP_HEIGHT] = 0

        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]

        contours, _ = cv2.findContours(img_binary_lp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by size and consider the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        char_images = []
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if dimensions[0] < w < dimensions[1] and dimensions[2] < h < dimensions[3]:
                bounding_boxes.append((x, y, w, h))
                
                char_copy = np.zeros((44, 24), dtype=np.uint8)
                char = img_binary_lp[y:y+h, x:x+w]
                char = cv2.resize(char, (20, 40))
                
                # Since char is already white text on black bg, we just place it in the center.
                # (Removed cv2.subtract to prevent polarity flipping)
                char_copy[2:42, 2:22] = char
                
                char_images.append(char_copy)

        # Group characters into lines based on Y coordinate
        # If two characters are within half a character height of each other in Y, they are on the same line.
        avg_h = np.mean([h for _, _, _, h in bounding_boxes]) if bounding_boxes else 0
        
        def get_line(y):
            return int(y // (avg_h * 0.75))

        # Sort by Line (Y-group) first, then by X-coordinate
        indices = sorted(range(len(bounding_boxes)), key=lambda k: (get_line(bounding_boxes[k][1]), bounding_boxes[k][0]))
        char_images_sorted = [char_images[idx] for idx in indices]
                
        return char_images_sorted

    def read_text(self, image):
        """
        Read text from a cropped license plate using HOG+SVM.
        Returns text, confidence, and the segmented character images.
        """
        if self.model is None:
            return "SVM_MISSING", 0.0, []
            
        char_images = self.segment_characters(image)
        if not char_images:
            return "", 0.0, []
            
        predictions = []
        for char_img in char_images:
            features = self.extract_hog_features(char_img)
            # SVM predict expects 2D array
            pred = self.model.predict([features])
            predictions.append(str(pred[0]))
            
        text = "".join(predictions)
        return text, 0.8, char_images

class RobustSVMReader(LegacySVMReader):
    def __init__(self, model_path='models/svm_model.pkl'):
        super().__init__(model_path)
        self.segmenter = ProjectionSegmenter(char_size=(44, 24))
        
    def segment_characters(self, image):
        """Override the legacy contour logic with robust projection profiling."""
        return self.segmenter.segment(image)

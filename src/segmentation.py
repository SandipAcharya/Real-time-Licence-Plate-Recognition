import cv2
import numpy as np

class ProjectionSegmenter:
    def __init__(self, char_size=(44, 24)):
        self.char_size = char_size

    def _normalize_binary(self, binary):
        """Ensure the background is black and text is white."""
        h, w = binary.shape
        # Look at the center 50% of the image
        center_region = binary[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        
        white_pixels = cv2.countNonZero(center_region)
        total_pixels = center_region.size
        
        if white_pixels > total_pixels / 2:
            return cv2.bitwise_not(binary)
        return binary

    def _preprocess(self, image):
        """Preprocess the image to binary."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Bilateral filter to reduce noise while keeping edges sharp
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Guarantee white text on black background
        binary = self._normalize_binary(binary)
        
        return binary

    def _find_lines(self, binary_img):
        """Find lines using horizontal projection profile."""
        horizontal_proj = np.sum(binary_img, axis=1)
        
        # Find peaks in horizontal projection
        # Threshold to ignore small noise
        threshold = np.max(horizontal_proj) * 0.1
        
        lines = []
        in_line = False
        start_y = 0
        
        for y, val in enumerate(horizontal_proj):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                # Filter out lines that are too small (noise)
                if (y - start_y) > 10: 
                    lines.append((start_y, y))
                    
        # If no lines found, assume the whole image is one line
        if not lines:
            lines.append((0, binary_img.shape[0]))
            
        return lines

    def _segment_chars_in_line(self, line_img):
        """Find characters using vertical projection profile."""
        vertical_proj = np.sum(line_img, axis=0)
        
        threshold = np.max(vertical_proj) * 0.05
        
        chars = []
        in_char = False
        start_x = 0
        
        for x, val in enumerate(vertical_proj):
            if val > threshold and not in_char:
                in_char = True
                start_x = x
            elif val <= threshold and in_char:
                in_char = False
                # Filter out very thin noise
                if (x - start_x) > 5:
                    chars.append((start_x, x))
                    
        # If no characters found, assume whole line is one character
        if not chars:
            chars.append((0, line_img.shape[1]))
            
        return chars

    def segment(self, image):
        """
        Segments the image into individual characters using projection profiling.
        Returns a list of clean, isolated character images resized to target dimensions.
        """
        binary = self._preprocess(image)
        
        # Clean up binary image with morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        lines = self._find_lines(binary)
        
        char_images = []
        
        for start_y, end_y in lines:
            line_img = binary[start_y:end_y, :]
            char_bounds = self._segment_chars_in_line(line_img)
            
            for start_x, end_x in char_bounds:
                char = binary[start_y:end_y, start_x:end_x]
                
                # Filter out extreme aspect ratios (probably noise)
                h, w = char.shape
                if h == 0 or w == 0: continue
                
                # Resize and format character to match legacy model expectations (44x24 canvas, 20x40 char)
                char_resized = cv2.resize(char, (20, 40))
                
                char_canvas = np.zeros(self.char_size, dtype=np.uint8)
                
                # Since char is already white text on black bg, we just place it in the center.
                # (Removed cv2.subtract to prevent polarity flipping)
                char_canvas[2:42, 2:22] = char_resized
                
                char_images.append(char_canvas)
                
        return char_images

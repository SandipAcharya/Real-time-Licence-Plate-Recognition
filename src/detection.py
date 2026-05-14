import cv2
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 model for License Plate Detection.
        For production, use a YOLOv8 model fine-tuned on Nepali license plates.
        """
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect license plates in an image.
        
        Args:
            image (numpy.ndarray): BGR image loaded via cv2.
            
        Returns:
            list of dicts containing 'box' (x1, y1, x2, y2), 'confidence', and 'class'.
            cropped_images (list of numpy.ndarray): The cropped license plate images.
        """
        # Perform inference
        results = self.model(image)[0]
        
        detections = []
        cropped_plates = []
        
        for box in results.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Extract the cropped plate
            cropped = image[y1:y2, x1:x2]
            
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls_id
            })
            cropped_plates.append(cropped)
            
        return detections, cropped_plates

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes on the image.
        """
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put text
            cv2.putText(img_copy, f"Plate: {conf:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return img_copy

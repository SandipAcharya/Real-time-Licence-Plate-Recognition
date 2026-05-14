import sqlite3
import datetime
import os
import cv2

class PlateDatabase:
    def __init__(self, db_path='logs/plates.db', images_dir='logs/images/'):
        """
        Initialize the SQLite database for logging detected license plates.
        """
        self.db_path = db_path
        self.images_dir = images_dir
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.init_db()

    def init_db(self):
        """Create the table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                ocr_method TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_detection(self, plate_text, confidence, plate_image, ocr_method="EasyOCR"):
        """
        Log a new detection to the database and save the cropped image.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp_str}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        
        # Save image
        if plate_image is not None and plate_image.size > 0:
            cv2.imwrite(image_path, plate_image)
        else:
            image_path = None
            
        # Insert into DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (plate_text, confidence, image_path, ocr_method)
            VALUES (?, ?, ?, ?)
        ''', (plate_text, confidence, image_path, ocr_method))
        conn.commit()
        conn.close()

    def get_all_logs(self):
        """Retrieve all detection logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

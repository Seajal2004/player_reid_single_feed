# src/detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, conf=0.3)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    # Based on model output: players are likely class 1 or 2
                    if int(cls) in [1, 2] and conf > 0.3:  
                        if (x2 - x1) * (y2 - y1) > 1000:
                            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        
        return detections
    
    def extract_features(self, frame, bbox):
        """Extract simple visual features from player bounding box"""
        x1, y1, x2, y2 = bbox[:4]
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return np.zeros(64)
            
        # Resize to standard size and extract color histogram
        roi_resized = cv2.resize(roi, (32, 64))
        hist_b = cv2.calcHist([roi_resized], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([roi_resized], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([roi_resized], [2], None, [16], [0, 256])
        
        features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten(), [x2-x1, y2-y1]])
        return features / (np.linalg.norm(features) + 1e-6)
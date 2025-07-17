import cv2
import numpy as np
from src.detector import PlayerDetector
from src.stable_tracker import StableTracker
import os

detector = PlayerDetector("models/best.pt")
tracker = StableTracker()

cap = cv2.VideoCapture("videos/15sec_input_720p.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter("output/tracked_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,128), (255,165,0)]
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)
    
    for x1, y1, x2, y2, conf, track_id in tracks:
        color = colors[int(track_id) % len(colors)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"Player {int(track_id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    out.write(frame)

cap.release()
out.release()
print("Output saved: output/tracked_video.mp4")
# src/stable_tracker.py - Stable ID tracker for consistent player identification
import numpy as np
import cv2
from scipy.spatial.distance import cosine

class StableTracker:
    def __init__(self):
        self.players = {}  # {id: {'features': [], 'last_seen': frame, 'bbox': []}}
        self.next_id = 1
        self.frame_count = 0
        self.similarity_threshold = 0.3  # Stricter matching
        self.max_missing_frames = 90  # Keep players much longer
        self.initial_assignment_done = False
        self.max_players = 15  # Limit total players
        
    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(64)
        
        roi = cv2.resize(roi, (32, 64))
        
        # Simple color features
        hist_b = cv2.calcHist([roi], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [16], [0, 256])
        
        features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        return features / (np.linalg.norm(features) + 1e-6)
    
    def compute_similarity(self, feat1, feat2):
        return 1 - cosine(feat1, feat2)
    
    def update(self, detections, frame):
        self.frame_count += 1
        current_tracks = []
        
        if len(detections) == 0:
            return current_tracks
        
        # Extract features for all detections
        detection_features = []
        for det in detections:
            features = self.extract_features(frame, det)
            detection_features.append(features)
        
        # Create similarity matrix for Hungarian assignment
        active_players = [pid for pid, pdata in self.players.items() 
                         if self.frame_count - pdata['last_seen'] <= self.max_missing_frames]
        
        if len(active_players) > 0:
            similarity_matrix = np.zeros((len(detections), len(active_players)))
            
            for i, det_features in enumerate(detection_features):
                for j, player_id in enumerate(active_players):
                    player_data = self.players[player_id]
                    if len(player_data['features']) > 0:
                        avg_features = np.mean(player_data['features'][-3:], axis=0)
                        similarity = self.compute_similarity(avg_features, det_features)
                        similarity_matrix[i][j] = similarity if similarity > self.similarity_threshold else 0
            
            # Simple greedy assignment (best matches first)
            used_detections = set()
            used_players = set()
            
            # Sort by similarity and assign
            assignments = []
            for i in range(len(detections)):
                for j in range(len(active_players)):
                    if similarity_matrix[i][j] > 0:
                        assignments.append((similarity_matrix[i][j], i, j))
            
            assignments.sort(reverse=True)  # Best matches first
            
            for similarity, det_idx, player_idx in assignments:
                if det_idx not in used_detections and player_idx not in used_players:
                    player_id = active_players[player_idx]
                    player_data = self.players[player_id]
                    
                    # Update player
                    player_data['features'].append(detection_features[det_idx])
                    player_data['last_seen'] = self.frame_count
                    player_data['bbox'] = detections[det_idx]
                    
                    if len(player_data['features']) > 8:
                        player_data['features'] = player_data['features'][-8:]
                    
                    current_tracks.append((*detections[det_idx], player_id))
                    used_detections.add(det_idx)
                    used_players.add(player_idx)
        else:
            used_detections = set()
        
        # Only create new players in first few frames or if under limit
        if self.frame_count <= 60 and len(self.players) < self.max_players:
            for i, det_features in enumerate(detection_features):
                if i not in used_detections:
                    new_id = self.next_id
                    self.next_id += 1
                    
                    self.players[new_id] = {
                        'features': [det_features],
                        'last_seen': self.frame_count,
                        'bbox': detections[i]
                    }
                    
                    current_tracks.append((*detections[i], new_id))
        
        return current_tracks
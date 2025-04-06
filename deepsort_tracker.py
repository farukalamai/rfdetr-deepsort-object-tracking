import numpy as np
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker

class DeepSORT:
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2):
        # Create a nearest neighbor distance metric
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance)
        # Create tracker
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        
    def update_with_detections(self, detections):
        """
        Update the tracker with new detections
        
        Parameters:
        -----------
        detections: sv.Detections
            Detections from the current frame
            
        Returns:
        --------
        sv.Detections
            Updated detections with tracker IDs
        """
        if len(detections) == 0:
            # No detections, return empty detections with tracker_id as None
            detections.tracker_id = np.array([], dtype=int)
            return detections
        
        # Convert supervision detections to DeepSORT detections
        deep_sort_detections = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            # Convert to tlwh (top-left width-height) format
            tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            confidence = detections.confidence[i]
            
            # For demonstration, we're using a simple feature vector
            # In a real implementation, you would extract meaningful appearance features
            # from the image region defined by the detection bbox
            feature = np.random.rand(128).astype(np.float32)
            
            deep_sort_detection = Detection(tlwh, confidence, feature)
            deep_sort_detections.append(deep_sort_detection)
        
        # Predict the next state of existing tracks
        self.tracker.predict()
        
        # Update tracker with new detections
        self.tracker.update(deep_sort_detections)
        
        # Initialize tracker_id array with -1 (no track)
        tracker_ids = np.full(len(detections), -1, dtype=int)
        
        # Collect active tracks (confirmed ones)
        active_tracks = [track for track in self.tracker.tracks if track.is_confirmed()]
        
        # For each active track, find the corresponding detection
        # This is a simplified approach - in a real implementation, you would
        # use the match results directly from the tracker's update method
        for track in active_tracks:
            # Get the track's current position
            track_tlwh = track.to_tlwh()
            track_tlbr = [
                track_tlwh[0], track_tlwh[1],
                track_tlwh[0] + track_tlwh[2], track_tlwh[1] + track_tlwh[3]
            ]
            
            # Find the detection with the highest IoU to this track
            best_match_idx = -1
            best_iou = 0
            
            for det_idx, bbox in enumerate(detections.xyxy):
                # Calculate IoU
                # Convert detection from xyxy (x1, y1, x2, y2) to tlbr format
                det_tlbr = [bbox[0], bbox[1], bbox[2], bbox[3]]
                
                # Calculate intersection
                x1 = max(track_tlbr[0], det_tlbr[0])
                y1 = max(track_tlbr[1], det_tlbr[1])
                x2 = min(track_tlbr[2], det_tlbr[2])
                y2 = min(track_tlbr[3], det_tlbr[3])
                
                if x2 < x1 or y2 < y1:
                    continue  # No intersection
                
                intersection = (x2 - x1) * (y2 - y1)
                
                # Calculate areas
                track_area = (track_tlbr[2] - track_tlbr[0]) * (track_tlbr[3] - track_tlbr[1])
                det_area = (det_tlbr[2] - det_tlbr[0]) * (det_tlbr[3] - det_tlbr[1])
                
                # Calculate IoU
                iou = intersection / (track_area + det_area - intersection)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = det_idx
            
            # If we found a good match, assign the track ID to the detection
            if best_match_idx >= 0 and best_iou > 0.5:  # Threshold can be adjusted
                tracker_ids[best_match_idx] = track.track_id
        
        # Assign tracker IDs to detections
        detections.tracker_id = tracker_ids
        return detections
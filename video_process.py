import supervision as sv
from rfdetr import RFDETRBase
from tqdm import tqdm
import numpy as np
from rfdetr.util.coco_classes import COCO_CLASSES
from deepsort_tracker import DeepSORT

# Create a DeepSORT tracker instance
deep_sort_tracker = DeepSORT(max_age=30, n_init=3)

model = RFDETRBase()
SOURCE_VIDEO_PATH = "854671-hd_1920_1080_25fps.mp4"
TARGET_VIDEO_PATH = "deepsort-tracking-result.mp4"

# Create a generator for video frames
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Process the video frame by frame
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, desc="Processing video"):
        # Get detections using RFDETRBase model
        detections = model.predict(frame, threshold=0.5)
        
        # Update tracker with new detections
        detections = deep_sort_tracker.update_with_detections(detections)
        
        # Create labels for all detections
        labels = [
            f"#{tracker_id if tracker_id != -1 else '?'} {COCO_CLASSES[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]
        
        # Annotate the frame
        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator(thickness=3).annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator(text_scale=1, text_thickness=2).annotate(annotated_frame, detections, labels)
        
        # Write the annotated frame to output video
        sink.write_frame(annotated_frame)

print(f"Video processing complete. Output saved to {TARGET_VIDEO_PATH}")
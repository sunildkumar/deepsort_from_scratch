import os

import cv2
import numpy as np

from constants import VIDEO_DIR
from detections import Model
from sort import Sort


def process_video(video_path: str, model: Model):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create side-by-side output video (twice the width)
    output_path = os.path.join(VIDEO_DIR, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

    # Initialize SORT tracker
    tracker = Sort(max_age=7, min_hits=3, iou_threshold=0.3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections from model
        detections, detection_frame = model.infer(frame)

        # Create tracking visualization frame
        tracking_frame = frame.copy()

        # Update tracker with new detections
        tracks = tracker.update(detections)

        # Draw tracking information
        for track in tracks:
            bbox = track.get_state()
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox[0] * frame_width)
            y1 = int(bbox[1] * frame_height)
            x2 = int(bbox[2] * frame_width)
            y2 = int(bbox[3] * frame_height)

            # Draw box and ID
            cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                tracking_frame,
                f"ID: {track.id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Add labels to both frames
        cv2.putText(
            detection_frame,
            "Detections",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            tracking_frame,
            "Tracking",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Combine frames side by side
        combined_frame = np.hstack((detection_frame, tracking_frame))

        out.write(combined_frame)

    cap.release()


if __name__ == "__main__":
    model = Model()
    video_path = os.path.join(VIDEO_DIR, "simple_traffic_video.mp4")
    process_video(video_path, model)

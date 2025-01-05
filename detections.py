import os

import cv2
import numpy as np
from imgcat import imgcat
from ultralytics import YOLO
from ultralytics.engine.results import Results

from constants import VIDEO_DIR


class Model:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.model.to("cuda")

    def infer(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """
        Returns:
            tuple containing:
            - list of detection dicts with bbox coords in xyxyn format
            - annotated image with boxes drawn
        """
        results: Results = self.model(image, verbose=False)[0]
        boxes = results.boxes.xyxyn.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        detections = []
        for box, class_id, score in zip(boxes, classes, scores):
            detections.append(
                {
                    "bbox": box,
                    "class_id": int(class_id),
                    "class_name": results.names[int(class_id)],
                    "confidence": float(score),
                }
            )

        annotated_frame = results.plot()
        return detections, annotated_frame


if __name__ == "__main__":
    model = Model()

    # load the first frame of the video
    video_path = os.path.join(VIDEO_DIR, "simple_traffic_video.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read video")
    boxes, annotated_frame = model.infer(frame)
    print(boxes)
    imgcat(annotated_frame)

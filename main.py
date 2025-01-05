import os

import cv2

from constants import VIDEO_DIR
from detections import Model


def process_video(video_path: str, model: Model):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(VIDEO_DIR, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections, annotated_frame = model.infer(frame)
        out.write(annotated_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    model = Model()
    video_path = os.path.join(VIDEO_DIR, "simple_traffic_video.mp4")
    process_video(video_path, model)

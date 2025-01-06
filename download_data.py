import os

import cv2
import yt_dlp

from constants import DATA_DIR, VIDEO_DIR

# This script downloads vidoes to use for testing our implementation.


def download_all_videos():
    # check that the data directory exists, if not create it
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # simple 30s video of cars driving down a city road. Narrow field of view and no occlusion.
    simple_traffic_video = (
        "https://www.youtube.com/watch?v=Gr0HpDM8Ki8&ab_channel=TechChannel00001"
    )

    # more complex video with many cars.
    complex_traffic_video = (
        "https://www.youtube.com/watch?v=MNn9qKG2UFI&ab_channel=KarolMajek"
    )

    # 100m dash race.
    race_video = "https://www.youtube.com/watch?v=3nbjhpcZ9_g&ab_channel=realsbstn"

    video_urls = {
        "simple_traffic_video": {"url": simple_traffic_video, "stop_time": 30},
        "complex_traffic_video": {"url": complex_traffic_video, "stop_time": None},
        "race_video": {"url": race_video, "stop_time": 15},
    }

    for video_name, video_info in video_urls.items():
        download_video(video_name, video_info["url"], video_info["stop_time"])


def download_video(video_name, video_url, stop_time=None):
    output_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    temp_path = os.path.join(VIDEO_DIR, f"{video_name}_full.mp4")

    if os.path.exists(output_path):
        print(f"Video already exists: {output_path}")
        return output_path

    try:
        # Download full video first
        ydl_opts = {
            "format": "best",
            "outtmpl": temp_path,
            "merge_output_format": "mp4",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video: {video_name} from {video_url}")
            ydl.extract_info(video_url, download=True)

        if stop_time is not None:
            # Clip the video using OpenCV
            cap = cv2.VideoCapture(temp_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(stop_time * fps)

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write frames until stop_time
            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            # Clean up
            cap.release()
            out.release()
            os.remove(temp_path)
            print(f"Clipped and saved video to: {output_path}")
        else:
            # If no stop_time, just rename the temp file
            os.rename(temp_path, output_path)
            print(f"Saved full video to: {output_path}")

        return output_path

    except Exception as e:
        print(f"Error processing video {video_url}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


if __name__ == "__main__":
    download_all_videos()

import os

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

    video_urls = {
        "simple_traffic_video": simple_traffic_video,
    }

    for video_name, video_url in video_urls.items():
        download_video(video_name, video_url)


def download_video(video_name, video_url):
    try:
        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": os.path.join(VIDEO_DIR, f"{video_name}.%(ext)s"),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video: {video_name} from {video_url}")
            info = ydl.extract_info(video_url, download=True)
            output_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
            print(f"Downloaded video: {video_name} to {output_path}")
            return output_path

    except Exception as e:
        print(f"Error downloading video {video_url}: {str(e)}")
        return None


if __name__ == "__main__":
    download_all_videos()

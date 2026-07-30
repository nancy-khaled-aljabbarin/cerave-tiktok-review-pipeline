from src.speech_to_text.video_downloader import download_video


video_url = (
    "https://www.tiktok.com/"
    "@aarinnola/video/7593777591492807958"
)

video_id = "test_video"

video_path = download_video(
    video_url=video_url,
    video_id=video_id,
)

print("Video downloaded successfully!")
print(f"Saved at: {video_path}")
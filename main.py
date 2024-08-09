from utilities import read_video, save_video
from trackers import PlayerTracker

def main():
    input_video_path = "input_videos/input_video.mp4"
    # Read the video frames
    video_frames = read_video(input_video_path)
    save_video(video_frames, "output_videos/output_video.avi")
    # Initialize the player tracker
    player_tracker = PlayerTracker("yolov8x.pt")
    player_detection = player_tracker.detect_frames(video_frames)
    save_video(video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
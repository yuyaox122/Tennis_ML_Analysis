from utilities import (read_video, 
                                       save_video)
from trackers import PlayerTracker, BallTracker
import cv2

def main():
    input_video_path = "input_videos/short_clip.mp4"
    # Read the video frames
    video_frames = read_video(input_video_path)

    # Initialize the player tracker and ball tracker
    player_tracker = PlayerTracker(model_path="C:/Users/yuyao/OneDrive/Documents/EPQ/models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="C:/Users/yuyao/OneDrive/Documents/EPQ/models/yolov5_last.pt")
    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="tracker_stubs/ball_detections.pkl")
    # Draw bounding boxes around the players and balls
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
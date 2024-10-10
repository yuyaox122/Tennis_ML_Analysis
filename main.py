from utilities import (read_video,
                       save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
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
    

    court_model_path = "C:/Users/yuyao/OneDrive/Documents/EPQ/models/court_points_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    player_detections = player_tracker.filter_players(court_keypoints, player_detections)
    mini_court = MiniCourt(video_frames[0])
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    
    player_mini_court_positions, ball_mini_court_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)
    # Draw bounding boxes around the players and balls
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    # Draw the key points on the court
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    # Draw the mini court  
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_positions)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_positions, color=(0, 255, 255))
    
    # Add frame numbers to the output video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        # Initialise the YOLO model with the given model path
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # Extract the ball positions from the dictionary
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Convert the list of dictionaries to a pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        # Interpolate the missing values in the DataFrame
        df_ball_positions = df_ball_positions.interpolate()
        # Fill the remaining missing values with the last valid value
        df_ball_positions = df_ball_positions.bfill()
        # Convert the DataFrame back to a list of dictionaries
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        # Extract the ball positions from the dictionary
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Convert the list into a pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Initialize the 'ball_hit' column with 0
        df_ball_positions['ball_hit'] = 0

        # Calculate the midpoint of the y-coordinates of the bounding boxes
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        # Calculate the rolling mean of the mid_y values with a window size of 5
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        # Calculate the difference between consecutive rolling mean values
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        # Define the minimum number of frames for a hit detection
        minimum_change_frames_for_hit = 25

        # Iterate through the DataFrame to detect ball hits
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            # Check for a negative to positive or positive to negative change in delta_y
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                # Check for consistent changes in the following frames
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                # Mark the frame as a ball hit if the change count exceeds the threshold
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        # Get the frame numbers where ball hits are detected
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        # If read_from_stub is True, load ball detections from the stub file
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Detect ball positions in each frame
        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        # Save ball detections to the stub file if stub_path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        # Use the YOLO model to predict ball positions in the frame
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        # Extract the bounding box coordinates for each detected ball
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        # Draw bounding boxes on each frame
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw the ball ID and bounding box on the frame
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):
    # Iterate over each row in player_stats DataFrame
    for index, row in player_stats.iterrows():
        # Extract player statistics from the current row
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        # Get the current frame from the output video frames
        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        # Define the dimensions and position of the stats overlay
        width = 350
        height = 230
        start_x = frame.shape[1] - 380
        start_y = frame.shape[0] - 250
        end_x = start_x + width
        end_y = start_y + height

        # Create a semi-transparent overlay for the stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        # Draw the player stats on the frame
        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw shot speed stats
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw player speed stats
        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw average shot speed stats
        text = "Mean Shot"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw average player speed stats
        text = "Mean Player"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_video_frames
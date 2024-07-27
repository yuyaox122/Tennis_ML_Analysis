import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Read until video is completed
    while cap.isOpened():
        # Return the frame and the flag ret
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames, output_path):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()
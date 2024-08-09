from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        player_detections = []
        for frame in frames:
            # Perform object detection on the frame
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        return player_detections
    
    def detect_frame(self, frame):
        # Perform object detection on the frame
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        player_dict = {}
        # Loop through the boxes in the frame
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]  
            object_cls_id = box.cls.to_list()[0]
            # Get the class name of the object
            object_cls_name = id_name_dict[object_cls_id]
            # Check if the object is a person
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

def draw_bboxes(self, video_frames, player_detections):
    output_video_frames = []
    for frame, player_dict in zip(video_frames, player_detections):
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            # Draw the bounding box on the frame
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
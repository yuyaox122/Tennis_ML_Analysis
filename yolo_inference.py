from ultralytics import YOLO

# Load pretrained model 
model = YOLO('models/yolo8_last.pt')

# Perform inference on a video
result = model.predict('input_videos/short_clip.mp4', conf=0.2, save=True)
print(result)
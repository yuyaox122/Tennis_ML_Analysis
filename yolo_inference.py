from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('input_videos/short_clip.mp4', save=True)
print(result)
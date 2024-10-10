import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        # Map model to CPU
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # Transform the image to a tensor and normalize it
            transforms.ToTensor(),
            # Normalize the image with mean and standard deviation values
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Convert the image from BGR to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply the transformations and add a batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Get the model's predictions
            outputs = self.model(image_tensor)
        # Convert the output tensor to a numpy array
        keypoints = outputs.squeeze().cpu().numpy()
        # Get the original image dimensions
        original_h, original_w = image.shape[:2]
        # Scale the keypoints to the original image dimensions
        print(keypoints[::2])
        print(keypoints[1::2])
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        
        return keypoints

    def draw_keypoints(self, image, keypoints):
    # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            # Get the x and y coordinates of the keypoint
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # Draw the keypoint index near the keypoint
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            # Draw a circle at the keypoint location
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        # Initialize a list to hold the output video frames
        output_video_frames = []
        # Iterate over each frame in the video
        for frame in video_frames:
            # Draw keypoints on the current frame
            frame = self.draw_keypoints(frame, keypoints)
            # Append the frame with keypoints to the output list
            output_video_frames.append(frame)
        return output_video_frames


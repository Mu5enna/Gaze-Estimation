import torch
import os
from torchvision import models, transforms
from PIL import Image
import sys
import cv2
import numpy as np

model_save_path = "model_path"

class GazeModel(torch.nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.base = models.resnet34(pretrained=True)
        in_features = self.base.fc.in_features

        self.base.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2))
        
    def forward(self, x):
        return self.base(x)
    
def yaw_pitch_to_vector(yaw, pitch):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)

    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.array([x, y, z])
        
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def crop_eyes_from_frame(img, img_size=224):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12, minSize=(20, 20))

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[-1]

        x_min = min(ex1, ex2)
        y_min = min(ey1, ey2)
        x_max = max(ex1 + ew1, ex2 + ew2)
        y_max = max(ey1 + eh1, ey2 + eh2)

        pad = int(0.2 * (y_max - y_min))
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(img.shape[1], x_max + pad)
        y_max = min(img.shape[0], y_max + pad)

        eye_crop = img[y_min:y_max, x_min:x_max]

        h, w = eye_crop.shape[:2]
        if h < img_size and w < img_size:
            delta_h, delta_w = img_size - h, img_size - w
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0,0,0]        
            eye_crop = cv2.copyMakeBorder(eye_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return eye_crop, x_max, x_min, y_max, y_min
    return None, 0, 0, 0, 0

#mean & std values of ImageNet dataset
mean_c, std_c = [0.485,0.456,0.406], [0.229,0.224,0.225]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_c, std=std_c)])

model = GazeModel()
device = torch.device("cpu")

checkpoint = {
    "model_state_dict":model.state_dict()}

if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
else:
    sys.exit(f"Model couldn't found at {model_save_path}")
    
model.eval()

cap = cv2.VideoCapture(0)

while cv2.getWindowProperty('Gaze Estimation (Eye Crop)', cv2.WND_PROP_VISIBLE) >= 0:
    ret, frame = cap.read()
    if not ret: break
    
    eye_img, x_max, x_min, y_max, y_min = crop_eyes_from_frame(frame)
    if eye_img is not None:
        
        img_pil =  Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
        with torch.no_grad():
            output = model(img_tensor)
            yaw, pitch = output[0].cpu().numpy()
            
        v = yaw_pitch_to_vector(yaw, pitch)
        h, w, _ = frame.shape
        start_x = int(x_min + ((x_max - x_min) / 2))
        start_y = int(y_min + ((y_max - y_min) / 2))
        dx, dy = int(-v[0] * 12000), int(-v[1] * 12000)
        
        cv2.arrowedLine(frame, (start_x, start_y), (start_x + dx, start_y + dy), (0, 0, 255), 3, tipLength=0.3)
    
    cv2.imshow("Gaze Estimation (Eye Crop)", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

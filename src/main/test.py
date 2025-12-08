from ultralytics import YOLO
import torch

model = YOLO('yolov5s.pt') # change to your .pt file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(next(model.parameters()).device)
#cuda:0
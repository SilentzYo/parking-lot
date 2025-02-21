import cv2
from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

im = 'test/traffic.jpeg'

results = model(im)
results.show()
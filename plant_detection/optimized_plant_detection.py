import torch
import cv2
import os
import numpy as np
from pathlib import Path

class OptimizedPlantDetector:
    def __init__(self, model_path=None, input_size=(320, 320)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.to(self.device).eval()
        self.input_size = input_size

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_size)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image, hsv

    def detect(self, image_path, save_output=True):
        image, _ = self.preprocess(image_path)
        results = self.model(image)
        if save_output:
            results.save(save_dir='runs/detect/optimized')
        return results

    def batch_detect(self, folder_path):
        image_paths = list(Path(folder_path).glob("*.jpg"))
        results = []
        for path in image_paths:
            result = self.detect(str(path), save_output=False)
            results.append(result)
        return results

    def export_onnx(self, filename='optimized_model.onnx'):
        dummy_input = torch.zeros((1, 3, *self.input_size)).to(self.device)
        torch.onnx.export(self.model.model, dummy_input, filename, opset_version=11)
        print(f"Exported model to {filename}")

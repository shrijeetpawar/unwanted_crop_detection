import os
import torch
from plant_detection.EnhancedEpd import EnhancedEpd

def main():
    # Initialize detector with YOLOv5 medium model
    detector = EnhancedEpd(model_type='yolov5m')
    
    # Run inference (update path to your dataset)
    results = detector.evaluate(dataset_path='plant_dataset/test/')
    
    print(f"Detection Results:")
    print(f"- mAP: {results['map']:.4f}")
    print(f"- AP: {results['ap']:.4f}")
    print(f"- Loss: {results['loss']:.4f}")

if __name__ == "__main__":
    main()

import os
import sys
import argparse

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plant_detection.EnhancedEpd import EnhancedEpd

def main():
    # Initialize detector
    detector = EnhancedEpd(model_type='yolov5m')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Plant Detection")
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--folder', help='Path to folder of images')
    args = parser.parse_args()

    if args.image:
        # Single image detection
        boxes, scores, labels = detector.detect_with_ensemble(args.image)
        print(f"Detected {len(boxes)} plants in {args.image}")
        print(f"Results saved to runs/detect/enhanced")
    elif args.folder:
        # Batch processing
        for img_file in os.listdir(args.folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.folder, img_file)
                boxes, _, _ = detector.detect_with_ensemble(img_path)
                print(f"Processed {img_file} - Found {len(boxes)} plants")
        print("✅ Batch processing complete")
    else:
        print("Please specify --image or --folder argument")

if __name__ == "__main__":
    main()

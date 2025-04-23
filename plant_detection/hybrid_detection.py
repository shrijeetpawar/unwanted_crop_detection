# plant_detection/hybrid_detection.py

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from plant_detection.EnhancedEpd import EnhancedEpd  # Assumes EnhancedEpd is in same package

def run_hybrid_detection(image_path, use_opencv_filter=True, use_local_model=True):
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="e6MKzlf5Q13T6Z3kNQkT"
    )

    image = cv2.imread(image_path)
    result = CLIENT.infer(image_path, model_id="weeds-nxe1w/1?confidence=0.5")
    
    if result is None or "predictions" not in result:
        print("❌ Roboflow inference failed.")
        return image
    
    print(f"🔍 Roboflow returned {len(result['predictions'])} predictions")
    final_detections = []
    
    for prediction in result["predictions"]:
        class_name = prediction["class"]
        confidence = prediction["confidence"]
        
        if confidence < 0.5:
            continue
        
        x, y, width, height = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
        x1 = max(x - width // 2, 0)
        y1 = max(y - height // 2, 0)
        x2 = min(x1 + width, image.shape[1])
        y2 = min(y1 + height, image.shape[0])
        roi = image[y1:y2, x1:x2]

        # Step 3A: OpenCV green filter
        if use_opencv_filter:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
            green_ratio = cv2.countNonZero(green_mask) / (roi.shape[0] * roi.shape[1])
            if green_ratio < 0.2:
                continue

        # Step 3B: Validate using local Mask R-CNN (optional)
        if use_local_model:
            detector = EnhancedEpd(model_type='maskrcnn', backend='pytorch')
            result_img = detector.detect_image(image_path)
            # Could be extended to crop-based prediction if needed

        # Step 4: Mark
        color = (0, 255, 0) if class_name == "crop" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        final_detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })

    cv2.imwrite("hybrid_result.jpg", image)
    print("✅ Hybrid detection complete. Result saved to hybrid_result.jpg")
    return image, final_detections

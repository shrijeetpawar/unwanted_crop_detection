# yolo_opencv_detector.py
import cv2
import numpy as np
import os


# Increase confidence threshold and NMS for weeds
class YoloCropWeedDetector:
    def __init__(self, weights_path, config_path, names_path=None, conf_threshold=0.6, nms_threshold=0.4):
        self.conf_threshold = conf_threshold  # Increased from 0.4
        self.nms_threshold = nms_threshold
        self.weed_class_id = 1  # Verify this matches your obj.names
        
        # Load model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
    def detect(self, image_path):
        # Load image
        frame = cv2.imread(image_path)
        height, width = frame.shape[:2]
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass with focus on weed class
        output_layers = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layers)
        
        # Post-process with higher confidence for weeds
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Higher confidence required for weeds
                if class_id == self.weed_class_id and confidence > self.conf_threshold * 1.2:
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    # ... rest of detection code ...




# class YoloCropWeedDetector:
#     def __init__(self, weights_path, config_path, names_path, conf_threshold=0.5, nms_threshold=0.4):
#         self.net = cv2.dnn.readNet(weights_path, config_path)
#         self.conf_threshold = conf_threshold
#         self.nms_threshold = nms_threshold

#         with open(names_path, "r") as f:
#             self.classes = [line.strip() for line in f.readlines()]
        
#         layer_names = self.net.getLayerNames()
#         self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

#     def detect(self, image_path):
#         image = cv2.imread(image_path)
#         height, width = image.shape[:2]

#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#         self.net.setInput(blob)
#         outputs = self.net.forward(self.output_layers)

#         boxes, confidences, class_ids = [], [], []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]  # For newer OpenCV versions
            indices = indices.flatten()

        for i in indices:
            x, y, w, h = boxes[i]
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0) if self.classes[class_ids[i]] == "crop" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

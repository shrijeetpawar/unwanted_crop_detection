"""
Enhanced Plant Detection with Accuracy Improvements
"""
import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from pytorch_grad_cam import GradCAM
# Need to install this package: pip install ensemble-boxes
from ensemble_boxes import weighted_boxes_fusion

class EnhancedPlantDataset(Dataset):
    # def __init__(self, dataset_path, transform=None):
    #     self.image_paths = []
    #     self.labels = []
    #     self.transform = transform
    #     self.class_weights = [1.0]  # Single class (plant)
    def __init__(self, dataset_path, model_type='yolov5m', backend='pytorch'):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.backend = backend
        self.input_size = (640, 640)
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = 0.25
        self.iou_thresh = 0.3
        self.ensemble_models = []

        if self.backend == 'pytorch':
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            self.model.to(self.device).eval()
        else:
            self._init_opencv_model()


        # Load images from train and valid directories
            split = 'train'  # Define a default value for split
            img_dir = os.path.join(self.dataset_path, split, 'images')
            img_dir = os.path.join(dataset_path, split, 'images')
            if not os.path.exists(img_dir):
                print(f"Warning: Image directory not found: {img_dir}")
                return
                
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(img_dir, fname)
                    self.image_paths.append(img_path)
                    self.labels.append(0)  # Single class
                    print(f"Added: {img_path} (class 0)")
        
        print(f"Loaded {len(self.image_paths)} images")
        if not self.image_paths:
            raise ValueError("No valid images found in dataset directory")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            width, height = image.shape[1], image.shape[2]  # For tensor
        else:
            width, height = image.size  # For PIL Image

        target = {
            'boxes': torch.tensor([[0, 0, width, height]], dtype=torch.float32),  # Full image box
            'labels': torch.tensor([label], dtype=torch.int64),
            'masks': torch.zeros((1, height, width), dtype=torch.uint8)
        }

        return image, target
        


        # image = Image.open(self.image_paths[idx]).convert('RGB')
        # label = self.labels[idx]
        
        # if self.transform:
        #     image = self.transform(image)
        
        # # Create target dictionary with placeholder values
        # target = {
        #     'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),  # [xmin, ymin, xmax, ymax]
        #     'labels': torch.tensor([label], dtype=torch.int64),
        #     'masks': torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.uint8)  # Dummy mask
        # }
        
        # # Return image, target, and image path to match expected format in training/evaluation
        # #return image, target, self.image_paths[idx]
        # return image, target

class EnhancedEpd:

    def __init__(self, model_type='yolov5m', backend='pytorch'):
        self.model_type = model_type
        self.backend = backend
        self.input_size = (640, 640)
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = 0.25
        self.iou_thresh = 0.3
        self.ensemble_models = []

        if self.backend == 'pytorch':
            # Force using Mask R-CNN for better plant detection
            #self.model = maskrcnn_resnet50_fpn(pretrained=True)
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            
            self.model.to(self.device).eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.metrics = {
                'ap': 0.0,
                'map': 0.0,
                'precision': [],
                'recall': [],
                'f1': []
            }
        else:
            self._init_opencv_model()


    def detect_from_contours(self, image_path, contours):
        # image = cv2.imread(image_path)
        # for contour in contours:
        #     # Process each contour for detection
        #     # This could involve creating bounding boxes or masks
        #     # and passing them to the model for prediction
        #     pass  # Implement detection logic here
        self.filename = image_path
        result = self.detect_image(self.filename)
        cv2.imshow("Result", result)




    def train(self, dataset_path, epochs=100):
        dataset = EnhancedPlantDataset(dataset_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, pin_memory=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                pin_memory=True, collate_fn=self._collate_fn)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=epochs)
        
        class_weights = torch.tensor(dataset.class_weights).to(self.device)
        
        best_map = 0.0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for images, targets in train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += losses.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            val_metrics = self.evaluate(val_loader)
            val_metrics['loss'] = avg_loss
            
            print(f"Epoch {epoch+1}/{epochs} | mAP: {val_metrics['map']:.4f} | Loss: {val_metrics['loss']:.4f}")
            
            if val_metrics['map'] > best_map:
                best_map = val_metrics['map']
                torch.save(self.model.state_dict(), f'best_{self.model_type}.pth')


    def _collate_fn(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def evaluate(self, data_loader=None, dataset_path=None):
        if data_loader is None:
            dataset = EnhancedPlantDataset(dataset_path, transform=self.transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in data_loader:
                # Convert images to appropriate format for detection models
                images = list(image.to(self.device) for image in images)
                
                # For evaluation, use the model in inference mode
                outputs = self.model(images)
                
                # Process outputs based on model type
                for i, output in enumerate(outputs):
                    scores = output['scores'].cpu().numpy()
                    preds = output['labels'].cpu().numpy()
                    
                    # Get ground truth labels from targets
                    true_labels = targets[i]['labels'].cpu().numpy()
                    
                    # Filter by confidence threshold
                    mask = scores >= self.conf_thresh
                    scores = scores[mask]
                    preds = preds[mask]
                    
                    all_scores.extend(scores)
                    all_preds.extend(preds)
                    all_labels.extend(true_labels)
        
        # Calculate metrics
        # ... rest of the method remains the same
        
        # Calculate metrics if we have predictions
        if len(all_preds) > 0 and len(all_labels) > 0:
            # Convert to binary classification for AP calculation if needed
            binary_preds = [1 if p > 0 else 0 for p in all_preds]
            binary_labels = [1 if l > 0 else 0 for l in all_labels]
            binary_scores = all_scores
            
            try:
                self.metrics['ap'] = average_precision_score(binary_labels, binary_scores)
            except:
                self.metrics['ap'] = 0.0
                
            self.metrics['map'] = self._calculate_map(all_preds, all_labels)
            
            try:
                precision, recall, _ = precision_recall_curve(binary_labels, binary_scores)
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['f1'] = 2 * (precision * recall) / (precision + recall + 1e-6)
            except:
                self.metrics['precision'] = []
                self.metrics['recall'] = []
                self.metrics['f1'] = []
        else:
            # No predictions were made
            self.metrics['ap'] = 0.0
            self.metrics['map'] = 0.0
            self.metrics['precision'] = []
            self.metrics['recall'] = []
            self.metrics['f1'] = []
        
        return {
            'map': self.metrics['map'],
            'ap': self.metrics['ap'],
            'loss': total_loss / max(1, len(data_loader))
        }

    def _calculate_map(self, preds, labels):
        # Handle case where classes might be missing
        unique_classes = sorted(set(labels).union(set(preds)))
        
        if not unique_classes:
            return 0.0
            
        aps = []
        for class_id in unique_classes:
            class_preds = [1 if p == class_id else 0 for p in preds]
            class_labels = [1 if l == class_id else 0 for l in labels]
            
            # Only calculate AP if we have both positive and negative examples
            if sum(class_labels) > 0 and sum(class_labels) < len(class_labels):
                try:
                    ap = average_precision_score(class_labels, class_preds)
                    aps.append(ap)
                except:
                    # Skip if there's an error calculating AP
                    pass
        
        # Return mean AP if we have values, otherwise 0
        return np.mean(aps) if aps else 0.0
    

    def detect_opencv(self, image_path):
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_thresh:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0) if self.classes[class_ids[i]] == "crop" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img


    def detect_image(self, image_path):
        if self.backend == 'opencv':
            return self.detect_opencv(image_path)
        else:
            return self.detect_with_ensemble(image_path)


    def detect_with_ensemble(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for model in self.ensemble_models:
            with torch.no_grad():
                outputs = model(img_tensor)
            
            if isinstance(outputs, list):
                boxes = outputs[0]['boxes'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                labels = outputs[0]['labels'].cpu().numpy()
            else:
                # Handle non-list outputs (e.g., for classification models)
                # For detection models, we expect a list format
                # This would need to be customized based on the actual output format
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                labels = outputs.argmax(dim=1).cpu().numpy()
                # Create dummy boxes if not available
                boxes = np.array([[0, 0, 1, 1]] * len(scores))
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        
        if not all_boxes or len(all_boxes[0]) == 0:
            return np.array([]), np.array([]), np.array([])
            
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=None, iou_thr=self.iou_thresh
        )
        
        return fused_boxes, fused_scores, fused_labels

    def optimize_thresholds(self, dataset_path):
        dataset = EnhancedPlantDataset(dataset_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.linspace(0.3, 0.7, 20):
            self.conf_thresh = thresh
            results = self.evaluate(loader)
            
            # Make sure we have F1 scores to evaluate
            if 'f1' in self.metrics and len(self.metrics['f1']) > 0:
                current_f1 = np.max(self.metrics['f1'])
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_thresh = thresh
        
        self.conf_thresh = best_thresh
        print(f"Optimized confidence threshold: {best_thresh:.2f}")

    def _init_opencv_model(self):
        base = "plant_detection/Crop_and_weed_detection-master/performing_detection/data"
        self.weights_path = f"{base}/weights/crop_weed_detection.weights"
        self.cfg_path = f"{base}/cfg/crop_weed.cfg"
        self.names_path = f"{base}/names/obj.names"
        with open(self.names_path, "r") as f:
            self.classes = f.read().splitlines()
        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]


if __name__ == "__main__":
    detector = EnhancedEpd(model_type='yolov5m')
    # detector.train('weeds.v2-release.yolov5-obb')
    detector.train('weeds.v2-release.yolov5-obb', epochs=3)
    detector.optimize_thresholds('weeds.v2-release.yolov5-obb/valid/images')
    results = detector.evaluate(dataset_path='plant-detection/weeds.v2-release.yolov5-obb/valid/images')
    print(f"Final mAP: {results['map']:.4f}")
"""Enhanced Plant Detection using Deep Learning (YOLOv5 / Mask R-CNN)."""

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
from sklearn.metrics import confusion_matrix, precision_recall_curve
from pytorch_grad_cam import GradCAM


class PlantDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for label, class_name in enumerate(['crop', 'weed']):
            class_dir = os.path.join(dataset_path, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Epd:
    def __init__(self, model_type='yolov5'):
        self.model_type = model_type
        self.input_size = (640, 640)
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'yolov5':
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.class_names = self.model.names
        elif model_type == 'maskrcnn':
            self.model = maskrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.metrics = {}

    def train(self, dataset_path, epochs=100):
        dataset = PlantDataset(dataset_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        def mixup_data(x, y, alpha=1.0):
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
            index = torch.randperm(x.size(0))
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience = 5
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images, y_a, y_b, lam = mixup_data(images, labels)
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    loss = sum(loss for loss in outputs.values())
                else:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        val_loss += sum(loss.item() for loss in outputs.values())
                    else:
                        val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), f"{self.model_type}_best_model.pth")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, dataset_path):
        dataset = PlantDataset(dataset_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, list):
                    preds = [1 if len(o['boxes']) > 0 else 0 for o in outputs]
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        self.metrics = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        self._plot_metrics()

    def _plot_metrics(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(self.metrics['confusion_matrix'], cmap='Blues')
        plt.title("Confusion Matrix")

        plt.subplot(132)
        plt.plot(self.metrics['recall'], self.metrics['precision'])
        plt.title("Precision-Recall Curve")

        plt.subplot(133)
        plt.plot(self.metrics['f1'])
        plt.title("F1 Score")

        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
        plt.close()

    def detect(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            pred = 1 if (isinstance(output, list) and len(output[0]['boxes']) > 0) else 0

        # Grad-CAM
        if self.model_type == 'yolov5':
            target_layer = self.model.model[-2]
        else:
            target_layer = self.model.backbone.body.layer4[-1]

        cam = GradCAM(model=self.model, target_layer=target_layer)
        grayscale_cam = cam(input_tensor=image_tensor, target_category=pred)

        self._visualize_detection(image, pred, grayscale_cam)
        return pred

    def detect_from_contours(self, image_path, contours):
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path).convert('RGB')
        predictions = []

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            crop = pil_image.crop((x, y, x + w, y + h))
            transformed = self.transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(transformed)

                pred = 0  # default to crop

                if self.model_type == 'yolov5':
                    # output is a list of detections per image
                    detections = output[0]  # Shape: [num_detections, 6]
                    for det in detections:
                        conf = det[4].item()
                        if conf > 0.5:
                            pred = 1  # weed
                            break

                elif isinstance(output, list):  # Mask R-CNN
                    pred = 1 if len(output[0]['boxes']) > 0 else 0

                else:  # Classifier (ResNet, etc.)
                    _, pred = torch.max(output, 1)
                    pred = pred.item()

            # Draw results
            center, radius = cv2.minEnclosingCircle(contour)
            center = tuple(map(int, center))
            radius = int(radius)
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)
            cv2.circle(image, center, radius, color, 2)
            predictions.append(pred)

        output_path = "classified_output.jpg"
        cv2.imwrite(output_path, image)
        print(f"✅ Detection complete. Output saved as {output_path}")
        return predictions



    def _visualize_detection(self, image, pred, heatmap):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image)
        ax1.set_title(f"Prediction: {'Crop' if pred == 0 else 'Weed'}")
        ax2.imshow(image)
        ax2.imshow(heatmap[0], cmap='jet', alpha=0.5)
        ax2.set_title("Grad-CAM")
        for ax in (ax1, ax2): ax.axis('off')
        plt.savefig('detection_result.png')
        plt.close()

    def generate_report_from_contours(self, image_path, contours):
        image = cv2.imread(image_path)
        total_area = image.shape[0] * image.shape[1]
        weed_area = sum([cv2.contourArea(cnt) for cnt in contours])
        weed_percentage = (weed_area / total_area) * 100

        suggestions = ["Minimal weeds, monitor regularly."]
        if weed_percentage > 20:
            suggestions = ["High weed density detected.", "Consider mechanical or chemical removal."]
        elif weed_percentage > 5:
            suggestions = ["Moderate weed presence. Manual removal recommended."]

        return {
            "total_area": f"{total_area} px²",
            "weed_area": f"{weed_area:.2f} px²",
            "weed_percentage": f"{weed_percentage:.2f}%",
            "suggestions": suggestions
        }


# Example usage
if __name__ == "__main__":
    detector = Epd(model_type='yolov5')  # or 'maskrcnn'
    detector.detect('abc.jpg')

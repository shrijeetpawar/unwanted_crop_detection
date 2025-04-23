
#!/usr/bin/env python
"""Enhanced Plant Detection GUI with Deep Learning Integration."""
import os
import json
import cv2
import numpy as np
from .EnhancedEpd import EnhancedEpd
from tkinter import Tk, filedialog
from .yolo_opencv_detector import YoloCropWeedDetector



class PlantDetectionGUI:
    """Main GUI for plant detection with interactive parameter adjustment."""
    def on_trackbar_change(self, val):
        """Callback for any trackbar change."""
        self.process_image()


    def __init__(self, image_filename=None, plant_detection=None):
        """Initialize GUI with default or provided parameters."""
        self.plant_detection = plant_detection
        self.setup_window_properties()
        self.initialize_default_parameters()
        self.load_or_set_defaults()
        self.setup_image(image_filename)
        self.deep_learning_mode = False  # Flag for DL integration

    def setup_window_properties(self):
        """Configure window properties and names."""
        self.window = 'Plant Detection'
        self.hsv_window = 'HSV Selection'
        self.dl_window = 'Deep Learning Settings'
        self.window_width = 1000
        self.window_height = 700
        self.hsv_window_loaded = False
        self.dl_window_loaded = False
        self.trackbars_initialized = False

    def initialize_default_parameters(self):
        """Set default processing parameters."""
        self.blur_amount = 5
        self.morph_amount = 5
        self.iterations = 1
        self.hsv_bounds = [[30, 20, 20], [90, 255, 255]]
        self.from_file = False
        self.dl_params = {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'model_type': 'yolov5'
        }

    def load_or_set_defaults(self):
        """Attempt to load parameters from file or use defaults."""
        try:
            directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
            with open(directory + "plant-detection_inputs.json", 'r') as f:
                inputs = json.load(f)
                self.blur_amount = inputs.get('blur', self.blur_amount)
                self.morph_amount = inputs.get('morph', self.morph_amount)
                self.iterations = inputs.get('iterations', self.iterations)
                self.hsv_bounds = [
                    [inputs['H'][0], inputs['S'][0], inputs['V'][0]],
                    [inputs['H'][1], inputs['S'][1], inputs['V'][1]]
                ]
                self.from_file = True
        except (IOError, KeyError, json.JSONDecodeError) as e:
            print(f"Using default parameters: {e}")

    def setup_image(self, image_filename):
        """Set up the initial image for processing."""
        if image_filename:
            self.filename = image_filename
        else:
            self.filename = self.select_image_file()
            if not self.filename:  # If no file selected, use default
                directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
                self.filename = directory + 'soil_image.jpg'

    def select_image_file(self):
        """Open file dialog to select an image."""
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        return file_path

    def resize_image(self, image):
        """Resize image maintaining aspect ratio to fit display window."""
        if image is None:
            return None
            
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > self.window_width / self.window_height:
            new_width = self.window_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.window_height
            new_width = int(new_height * aspect_ratio)
            
        return cv2.resize(image, (new_width, new_height), 
                         interpolation=cv2.INTER_AREA)

            # In GUI.py - Update the process_image() method:
    def process_image(self):
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Check mode: traditional or deep learning
            if cv2.getTrackbarPos('DL Mode', self.window) == 1:
                # Deep learning mode
                weights_path = os.path.join(os.path.dirname(__file__), 
                                "Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights")
                config_path = os.path.join(os.path.dirname(__file__), 
                                "Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg")
                names_path = os.path.join(os.path.dirname(__file__), 
                    "Crop_and_weed_detection-master/performing_detection/data/names/obj.names")

                detector = YoloCropWeedDetector(weights_path=weights_path, config_path=config_path, names_path=names_path, conf_threshold=0.5)
                result = detector.detect(self.filename)
                self.display_result(result)

            else:
                # Traditional processing mode
                image = cv2.imread(self.filename)
                if image is None:
                    raise ValueError("Failed to load image")

                # Blur
                blur_img = cv2.GaussianBlur(image, (params['blur']*2+1, params['blur']*2+1), 0)
                # cv2.imshow('Blurred Image', self.resize_image(blur_img))

                # Convert to HSV and threshold
                hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
                lower = np.array(params['HSV_min'])
                upper = np.array(params['HSV_max'])
                mask = cv2.inRange(hsv_img, lower, upper)
                # cv2.imshow('HSV Threshold', self.resize_image(mask))

                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph'], params['morph']))
                morph_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=params['iterations'])
                # cv2.imshow('Morphological Opening', self.resize_image(morph_img))

                # Find contours (weed detection)
                contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result_img = image.copy()
                cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
                self.display_result(result_img)

                # Optionally save intermediate images
                cv2.imwrite('blurred_image.jpg', blur_img)
                cv2.imwrite('hsv_threshold.jpg', mask)
                cv2.imwrite('morphological_opening.jpg', morph_img)

        except Exception as e:
            print(f"Error: {str(e)}")
            self.show_error_message("Detection failed - showing original image")
            self.display_result(cv2.imread(self.filename))

        # """Process the image with current parameters."""
        # try:
        #     # Get current parameter values
        #     params = self.get_current_parameters()
            
        #     if self.deep_learning_mode:
        #         self.process_with_dl(params)
        #     else:
        #         self.process_traditional(params)
                
        # except Exception as e:
        #     print(f"Image processing error: {e}")
        #     self.show_error_message(str(e))

    def get_current_parameters(self):
        """Get current values from trackbars or defaults."""
        params = {
            'blur': self.blur_amount,
            'morph': self.morph_amount,
            'iterations': self.iterations,
            'HSV_min': self.hsv_bounds[0],
            'HSV_max': self.hsv_bounds[1]
        }
        
        if self.trackbars_initialized:
            try:
                params.update({
                    'blur': cv2.getTrackbarPos('Blur', self.window),
                    'morph': cv2.getTrackbarPos('Morph', self.window),
                    'iterations': cv2.getTrackbarPos('Iterations', self.window)
                })
            except cv2.error:
                pass
                
        if self.hsv_window_loaded:
            self._get_hsv_values()
            
        return params

    def process_traditional(self, params):
        """Process image using traditional CV methods."""
        plantdetection = self.plant_detection(
            image=self.filename,
            blur=params['blur'],
            morph=params['morph'],
            iterations=params['iterations'],
            HSV_min=params['HSV_min'],
            HSV_max=params['HSV_max'],
            GUI=True
        )
        plantdetection.detect_plants()
        self.display_result(plantdetection.final_marked_image)

    def process_with_dl(self, params):
        """Process image using deep learning (YOLOv5 or ResNet)."""
        self.detector.detect_from_contours(self.filename, contours)

        # Step 1: Use traditional pipeline to extract contours
        plantdetection = self.plant_detection(
            image=self.filename,
            blur=params['blur'],
            morph=params['morph'],
            iterations=params['iterations'],
            HSV_min=params['HSV_min'],
            HSV_max=params['HSV_max'],
            GUI=True
        )
        plantdetection.detect_plants()

        # Step 2: Extract contours from processed image
        contours = plantdetection.image.detected_contours  # Requires Image.py update as discussed

        # Step 3: Run DL classification on contours
        self.detector = EnhancedEpd(model_type='yolov5', backend='opencv')  # 'yolov5' or 'resnet'
        self.detector.detect_from_contours(self.filename, contours)

        # Step 4: Load and display result image
        result_img = cv2.imread("classified_output.jpg")  # or any output path from your DL script
        self.display_result(result_img)


    def display_result(self, image):
        """Display the processed image in the main window."""
        img_resized = self.resize_image(image)
        if img_resized is not None:
            cv2.imshow(self.window, img_resized)

    def show_error_message(self, message):
        """Display error message to user."""
        # Could be enhanced with a proper GUI dialog
        print(f"ERROR: {message}")

    def create_main_window(self):
        """Create and configure the main application window."""
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.window_width, self.window_height)
        
        # Create trackbars
        cv2.createTrackbar('Blur', self.window, self.blur_amount, 100, self.on_trackbar_change)
        cv2.createTrackbar('Morph', self.window, self.morph_amount, 100, self.on_trackbar_change)
        cv2.createTrackbar('Iterations', self.window, self.iterations, 100, self.on_trackbar_change)
        cv2.createTrackbar('Open HSV Window', self.window, 0, 1, self.on_trackbar_change)
        cv2.createTrackbar('DL Mode', self.window, 0, 1, self.on_trackbar_change)
        
        self.trackbars_initialized = True
        self.process_image()

    def run(self):
        """Run the main application loop."""
        self.create_main_window()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save_current_settings()
            elif key == ord('l'):
                self.load_new_image()
                
        cv2.destroyAllWindows()

    def save_current_settings(self):
        """Save current parameters to file."""
        settings = {
            'blur': cv2.getTrackbarPos('Blur', self.window),
            'morph': cv2.getTrackbarPos('Morph', self.window),
            'iterations': cv2.getTrackbarPos('Iterations', self.window),
            'H': [self.hsv_bounds[0][0], self.hsv_bounds[1][0]],
            'S': [self.hsv_bounds[0][1], self.hsv_bounds[1][1]],
            'V': [self.hsv_bounds[0][2], self.hsv_bounds[1][2]]
        }
        
        try:
            directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
            with open(directory + "plant-detection_inputs.json", 'w') as f:
                json.dump(settings, f)
            print("Settings saved successfully")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def load_new_image(self):
        """Load a new image file."""
        new_file = self.select_image_file()
        if new_file:
            self.filename = new_file
            self.process_image()


class CalibrationGUI:
    """Enhanced Calibration GUI with additional features."""
    
    def __init__(self, cimage_filename=None, image_filename=None, plant_detection=None):
        """Initialize calibration GUI."""
        self.plant_detection = plant_detection
        self.setup_window_properties()
        self.initialize_default_parameters()
        self.setup_images(cimage_filename, image_filename)
        self.trackbars_initialized = False

    def setup_window_properties(self):
        """Configure window properties."""
        self.window = 'Calibration'
        self.result_window = 'Calibration Result'
        self.window_width = 900
        self.window_height = 600
        self.result_width = 900
        self.result_height = 600

    def initialize_default_parameters(self):
        """Set default calibration parameters."""
        self.parameters = {
            'axis': 1,
            'origin_vert': 1,
            'origin_horiz': 0,
            'separation': 100,
            'offset_x': 50,
            'offset_y': 100,
            'iterations': 3
        }

    def setup_images(self, cimage_filename, image_filename):
        """Set up calibration and test images."""
        directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.cfilename = cimage_filename or directory + 'p2c_test_calibration.jpg'
        self.filename = image_filename or directory + 'soil_image.jpg'

    def run(self):
        """Run the calibration GUI."""
        self.create_windows()
        self.create_trackbars()
        self.process_image()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
                
        cv2.destroyAllWindows()

    # ... [Additional methods similar to PlantDetectionGUI but for calibration]
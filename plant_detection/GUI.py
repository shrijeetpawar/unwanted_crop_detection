#!/usr/bin/env python
"""Enhanced Plant Detection GUI with Deep Learning Integration and Color Coding."""
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

    def on_category_change(self, val):
        """Callback for category trackbar change."""
        self.update_color_preview()
        self.process_image()
    
    def update_color_preview(self):
        """Update the color preview window."""
        if not self.trackbars_initialized:
            return
            
        # Create color preview image
        preview = np.zeros((100, 300, 3), dtype=np.uint8)
        category_idx = cv2.getTrackbarPos('Category', self.window)
        category_names = list(self.color_settings.keys())
        
        # Draw color swatches for all categories
        for i, (cat, color) in enumerate(self.color_settings.items()):
            x_start = i * 100
            preview[:, x_start:x_start+100] = color
            # Highlight selected category
            if i == category_idx:
                cv2.rectangle(preview, (x_start, 0), (x_start+99, 99), (255, 255, 255), 3)
            cv2.putText(preview, cat, (x_start+10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow('Color Preview', preview)
        
    def __init__(self, image_filename=None, plant_detection=None):
        """Initialize GUI with default or provided parameters."""
        self.plant_detection = plant_detection
        self.setup_window_properties()
        self.initialize_default_parameters()
        self.load_or_set_defaults()
        self.setup_image(image_filename)
        self.deep_learning_mode = False  # Flag for DL integration
        
        # Add color settings for different plant categories
        self.color_settings = {
            'crop': (0, 255, 0),    # Green for crops
            'weed': (0, 0, 255),    # Red for weeds
            'grass': (255, 255, 0)  # Yellow for grass
        }

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
        # Category-specific HSV bounds
        self.category_hsv_bounds = {
            'crop': [[30, 40, 40], [90, 255, 255]],
            'weed': [[0, 30, 30], [30, 255, 255]],
            'grass': [[90, 20, 20], [140, 255, 255]]
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
                # Try to load category-specific settings
                if 'categories' in inputs:
                    for category, bounds in inputs['categories'].items():
                        if category in self.category_hsv_bounds:
                            self.category_hsv_bounds[category] = bounds
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

    def process_image(self):
        """Process the image with current parameters and color coding."""
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Check mode: traditional or deep learning
            if self.trackbars_initialized and cv2.getTrackbarPos('DL Mode', self.window) == 1:
                # Deep learning mode
                weights_path = os.path.join(os.path.dirname(__file__), 
                                "Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights")
                config_path = os.path.join(os.path.dirname(__file__), 
                                "Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg")
                names_path = os.path.join(os.path.dirname(__file__), 
                    "Crop_and_weed_detection-master/performing_detection/data/names/obj.names")

                detector = YoloCropWeedDetector(weights_path=weights_path, config_path=config_path, 
                                               names_path=names_path, conf_threshold=0.5)
                result = detector.detect(self.filename)
                self.display_result(result)

            else:
                # Traditional processing mode
                image = cv2.imread(self.filename)
                if image is None:
                    raise ValueError("Failed to load image")
                    
                # Create a copy of the original image for result visualization
                result_img = image.copy()
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Get currently selected category
                selected_category = 0  # Default to first category
                if self.trackbars_initialized:
                    selected_category = cv2.getTrackbarPos('Category', self.window)
                
                category_names = list(self.color_settings.keys())
                selected_name = category_names[selected_category] if selected_category < len(category_names) else category_names[0]
                
                # Process each category with different colors
                categories = {}
                for cat_name, color in self.color_settings.items():
                    categories[cat_name] = {
                        'hsv_min': self.category_hsv_bounds[cat_name][0],
                        'hsv_max': self.category_hsv_bounds[cat_name][1],
                        'color': color
                    }
                
                # For the currently selected category, use the trackbar values
                if selected_name in categories:
                    categories[selected_name]['hsv_min'] = params['HSV_min']
                    categories[selected_name]['hsv_max'] = params['HSV_max']
                    # Update the category's HSV bounds
                    self.category_hsv_bounds[selected_name] = [params['HSV_min'], params['HSV_max']]
                
                # Create an overlay mask for visualization
                overlay = np.zeros_like(image)
                
                # Process each category
                for category, cat_params in categories.items():
                    # Apply blur
                    blur_img = cv2.GaussianBlur(hsv_img, (params['blur']*2+1, params['blur']*2+1), 0)
                    
                    # Apply HSV thresholding
                    lower = np.array(cat_params['hsv_min'])
                    upper = np.array(cat_params['hsv_max'])
                    mask = cv2.inRange(blur_img, lower, upper)
                    
                    # Morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph'], params['morph']))
                    morph_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=params['iterations'])
                    
                    # Find contours
                    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours with category-specific color
                    cv2.drawContours(result_img, contours, -1, cat_params['color'], 2)
                    
                    # Create colored mask for this category
                    category_mask = np.zeros_like(image)
                    cv2.drawContours(category_mask, contours, -1, cat_params['color'], -1)  # -1 fills the contours
                    
                    # Add this category's mask to the overlay
                    overlay = cv2.addWeighted(overlay, 1, category_mask, 0.5, 0)
                
                # Blend the original image with the overlay
                final_result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                
                # Add contour outlines
                final_result = cv2.addWeighted(final_result, 1, result_img, 0.5, 0)
                
                self.display_result(final_result)

        except Exception as e:
            print(f"Error: {str(e)}")
            self.show_error_message("Detection failed - showing original image")
            self.display_result(cv2.imread(self.filename))

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

    def _get_hsv_values(self):
        """Get HSV values from HSV window trackbars."""
        if self.hsv_window_loaded:
            try:
                self.hsv_bounds = [
                    [cv2.getTrackbarPos('H min', self.hsv_window),
                     cv2.getTrackbarPos('S min', self.hsv_window),
                     cv2.getTrackbarPos('V min', self.hsv_window)],
                    [cv2.getTrackbarPos('H max', self.hsv_window),
                     cv2.getTrackbarPos('S max', self.hsv_window),
                     cv2.getTrackbarPos('V max', self.hsv_window)]
                ]
            except cv2.error:
                # HSV window might have been closed
                self.hsv_window_loaded = False

    def create_hsv_window(self):
        """Create and configure the HSV selection window."""
        cv2.namedWindow(self.hsv_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.hsv_window, 600, 400)
        
        # Create HSV trackbars
        cv2.createTrackbar('H min', self.hsv_window, self.hsv_bounds[0][0], 179, self.on_trackbar_change)
        cv2.createTrackbar('S min', self.hsv_window, self.hsv_bounds[0][1], 255, self.on_trackbar_change)
        cv2.createTrackbar('V min', self.hsv_window, self.hsv_bounds[0][2], 255, self.on_trackbar_change)
        cv2.createTrackbar('H max', self.hsv_window, self.hsv_bounds[1][0], 179, self.on_trackbar_change)
        cv2.createTrackbar('S max', self.hsv_window, self.hsv_bounds[1][1], 255, self.on_trackbar_change)
        cv2.createTrackbar('V max', self.hsv_window, self.hsv_bounds[1][2], 255, self.on_trackbar_change)
        
        self.hsv_window_loaded = True

    def display_result(self, image):
        """Display the processed image in the main window with legend."""
        if image is None:
            return
            
        # Add legend
        h, w = image.shape[:2]
        legend_height = 30 * len(self.color_settings)
        legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255  # White background
        
        for i, (name, color) in enumerate(self.color_settings.items()):
            y_pos = 20 + 30 * i
            # Draw color swatch
            cv2.rectangle(legend, (10, y_pos-15), (40, y_pos+5), color, -1)
            # Draw category name
            cv2.putText(legend, name, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Combine image and legend
        result_with_legend = np.vstack([image, legend])
        
        # Resize and display
        img_resized = self.resize_image(result_with_legend)
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
        cv2.createTrackbar('Open HSV Window', self.window, 0, 1, self.on_hsv_window_toggle)
        cv2.createTrackbar('DL Mode', self.window, 0, 1, self.on_trackbar_change)
        
        # Add category selection trackbar (0=crop, 1=weed, 2=grass)
        cv2.createTrackbar('Category', self.window, 0, 2, self.on_category_change)
        
        self.trackbars_initialized = True
        self.process_image()
        
        # Create color preview window
        self.update_color_preview()

    def on_hsv_window_toggle(self, val):
        """Toggle HSV window visibility."""
        if val == 1 and not self.hsv_window_loaded:
            self.create_hsv_window()
        elif val == 0 and self.hsv_window_loaded:
            cv2.destroyWindow(self.hsv_window)
            self.hsv_window_loaded = False
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
            'blur': self.blur_amount if not self.trackbars_initialized else cv2.getTrackbarPos('Blur', self.window),
            'morph': self.morph_amount if not self.trackbars_initialized else cv2.getTrackbarPos('Morph', self.window),
            'iterations': self.iterations if not self.trackbars_initialized else cv2.getTrackbarPos('Iterations', self.window),
            'H': [self.hsv_bounds[0][0], self.hsv_bounds[1][0]],
            'S': [self.hsv_bounds[0][1], self.hsv_bounds[1][1]],
            'V': [self.hsv_bounds[0][2], self.hsv_bounds[1][2]],
            'categories': {}
        }
        
        # Save category-specific HSV bounds
        for category, bounds in self.category_hsv_bounds.items():
            settings['categories'][category] = bounds
        
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

    def on_trackbar_change(self, val):
        """Callback for any trackbar change."""
        self.process_calibration()

    def create_window(self):
        """Create and configure the calibration window."""
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.window_width, self.window_height)
        
        # Create calibration trackbars
        cv2.createTrackbar('Axis', self.window, self.parameters['axis'], 1, self.on_trackbar_change)
        cv2.createTrackbar('Origin Vertical', self.window, self.parameters['origin_vert'], 1, self.on_trackbar_change)
        cv2.createTrackbar('Origin Horizontal', self.window, self.parameters['origin_horiz'], 1, self.on_trackbar_change)
        cv2.createTrackbar('Separation', self.window, self.parameters['separation'], 1000, self.on_trackbar_change)
        cv2.createTrackbar('Offset X', self.window, self.parameters['offset_x'], 1000, self.on_trackbar_change)
        cv2.createTrackbar('Offset Y', self.window, self.parameters['offset_y'], 1000, self.on_trackbar_change)
        cv2.createTrackbar('Iterations', self.window, self.parameters['iterations'], 10, self.on_trackbar_change)
        
        self.trackbars_initialized = True

    def process_calibration(self):
        """Process calibration with current parameters."""
        if not self.trackbars_initialized:
            return
            
        try:
            # Get current parameters from trackbars
            self.parameters = {
                'axis': cv2.getTrackbarPos('Axis', self.window),
                'origin_vert': cv2.getTrackbarPos('Origin Vertical', self.window),
                'origin_horiz': cv2.getTrackbarPos('Origin Horizontal', self.window),
                'separation': cv2.getTrackbarPos('Separation', self.window),
                'offset_x': cv2.getTrackbarPos('Offset X', self.window),
                'offset_y': cv2.getTrackbarPos('Offset Y', self.window),
                'iterations': cv2.getTrackbarPos('Iterations', self.window)
            }
            
            # Load calibration image
            calibration_img = cv2.imread(self.cfilename)
            if calibration_img is None:
                raise ValueError("Failed to load calibration image")
                
            # Display calibration image with parameters
            result_img = calibration_img.copy()
            
            # Draw calibration grid based on parameters
            height, width = calibration_img.shape[:2]
            
            # Draw reference lines
            axis_color = (0, 255, 0)  # Green for axis
            grid_color = (255, 0, 0)   # Blue for grid
            
            # Origin point
            origin_x = self.parameters['offset_x']
            origin_y = self.parameters['offset_y']
            if self.parameters['origin_horiz'] == 1:
                origin_x = width - origin_x
            if self.parameters['origin_vert'] == 1:
                origin_y = height - origin_y
                
            # Draw origin
            cv2.circle(result_img, (origin_x, origin_y), 10, (0, 0, 255), -1)
            
            # Draw axis lines
            if self.parameters['axis'] == 0:  # Horizontal axis
                cv2.line(result_img, (0, origin_y), (width, origin_y), axis_color, 2)
                
                # Draw vertical grid lines
                sep = self.parameters['separation']
                for i in range(1, int(width / sep) + 1):
                    x = origin_x + i * sep
                    if x < width:
                        cv2.line(result_img, (x, 0), (x, height), grid_color, 1)
                    x = origin_x - i * sep
                    if x >= 0:
                        cv2.line(result_img, (x, 0), (x, height), grid_color, 1)
            else:  # Vertical axis
                cv2.line(result_img, (origin_x, 0), (origin_x, height), axis_color, 2)
                
                # Draw horizontal grid lines
                sep = self.parameters['separation']
                for i in range(1, int(height / sep) + 1):
                    y = origin_y + i * sep
                    if y < height:
                        cv2.line(result_img, (0, y), (width, y), grid_color, 1)
                    y = origin_y - i * sep
                    if y >= 0:
                        cv2.line(result_img, (0, y), (width, y), grid_color, 1)
            
            # Display result
            cv2.imshow(self.window, result_img)
            
            # Display calibration information
            info_img = np.ones((200, width, 3), dtype=np.uint8) * 255
            info_text = [
                f"Axis: {'Vertical' if self.parameters['axis'] == 1 else 'Horizontal'}",
                f"Origin: {'Bottom' if self.parameters['origin_vert'] == 1 else 'Top'}-{'Right' if self.parameters['origin_horiz'] == 1 else 'Left'}",
                f"Separation: {self.parameters['separation']} pixels",
                f"Offset X: {self.parameters['offset_x']} pixels",
                f"Offset Y: {self.parameters['offset_y']} pixels"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(info_img, text, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
            cv2.imshow('Calibration Info', info_img)
            
        except Exception as e:
            print(f"Calibration error: {e}")

    def run(self):
        """Run the calibration GUI."""
        self.create_window()
        self.process_calibration()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save_calibration()
                
        cv2.destroyAllWindows()

    def save_calibration(self):
        """Save calibration parameters to file."""
        try:
            directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
            with open(directory + "calibration_params.json", 'w') as f:
                json.dump(self.parameters, f)
            print("Calibration parameters saved successfully")
        except Exception as e:
            print(f"Failed to save calibration: {e}")
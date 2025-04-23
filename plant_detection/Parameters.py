#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import os
import copy
import json
import cv2
import numpy as np
from plant_detection import ENV

class Parameters(object):
    """Input parameters for Plant Detection with multi-category support."""
    
    # Color schemes for visualization
    COLOR_SCHEME = {
        'blur': (255, 255, 0),    # Cyan
        'morph': (255, 0, 255),   # Magenta 
        'iterations': (0, 255, 255) # Yellow
    }
    
    # Colors for detection display
    DETECTION_COLORS = {
        'crop': (0, 255, 0),      # Bright Green
        'weed': (0, 0, 255),      # Pure Red
        'grass': (255, 255, 255)  # Clean White
    }
    
    # Size constraints (mm²)
    SIZE_LIMITS = {
        'crop': (500, 5000),
        'weed': (100, 2000),
        'grass': (50, 1000)
    }
    
    # Edge preservation parameters (sigma_s, sigma_r)
    EDGE_PRESERVATION = {
        'low': (30, 0.3),
        'medium': (50, 0.4),
        'high': (80, 0.5)
    }
    
    def __init__(self):
        # Category-specific parameters
        self.categories = {
            'crop': {
                'blur': 5, 'morph': 5, 'iterations': 2,
                'H': [40, 80], 'S': [50, 255], 'V': [50, 255],
                'color': (0, 255, 0),  # Green
                'active': True,
                'min_size': 500, 'max_size': 5000,
                'adaptive_s': False, 'adaptive_v': False,
                'edge_preservation': 'medium'
            },
            'weed': {
                'blur': 15, 'morph': 10, 'iterations': 3,
                'H': [[0, 10], [160, 180]], 'S': [50, 255], 'V': [50, 255],
                'color': (0, 0, 255),  # Red
                'active': True,
                'min_size': 100, 'max_size': 2000,
                'adaptive_s': True, 'adaptive_v': False,
                'edge_preservation': 'high'
            },
            'grass': {
                'blur': 3, 'morph': 3, 'iterations': 1,
                'H': [0, 180], 'S': [0, 50], 'V': [200, 255],
                'color': (255, 255, 255),  # White
                'active': False,
                'min_size': 50, 'max_size': 1000,
                'adaptive_s': False, 'adaptive_v': True,
                'edge_preservation': 'low'
            }
        }
        
        # Legacy parameters for backward compatibility
        self.legacy_defaults = {
            'blur': 15, 'morph': 10, 'iterations': 2,
            'H': [30, 90], 'S': [50, 255], 'V': [50, 255],
            'save_detected_plants': False,
            'use_bounds': True, 'min_radius': 1.5, 'max_radius': 50
        }
        
        # Calibration defaults
        self.calibration_defaults = {
            'blur': 15, 'morph': 6, 'iterations': 4,
            'H': [30, 90], 'S': [20, 255], 'V': [20, 255],
            'calibration_circles_xaxis': True,
            'image_bot_origin_location': [0, 1],
            'calibration_circle_separation': 100,
            'camera_offset_coordinates': [50, 100],
            'calibration_iters': 3,
            'total_rotation_angle': 0,
            'invert_hue_selection': True,
            'easy_calibration': False
        }

        # Detection priority order
        self.detection_order = ['crop', 'weed', 'grass']
        self.current_mode = 'detection'  # or 'calibration'
        
        # Processing parameters
        self.kernel_type = 'ellipse'
        self.morph_type = 'close'
        
        # Create dictionaries of morph types
        self.cv2_kt = {
            'ellipse': cv2.MORPH_ELLIPSE,
            'rect': cv2.MORPH_RECT,
            'cross': cv2.MORPH_CROSS
        }
        
        self.cv2_mt = {
            'close': cv2.MORPH_CLOSE,
            'open': cv2.MORPH_OPEN,
            'erode': 'erode',
            'dilate': 'dilate'
        }
        
        # Runtime values
        self.current_threshold = 0
        self.sigma_s = 50
        self.sigma_r = 0.4

    def save_to_env_var(self, widget):
        """Save parameters with category prefixes."""
        prefix = 'WEED_DETECTOR_'
        if 'calibration' in widget:
            prefix = 'CAMERA_CALIBRATION_'
            
        for category, params in self.categories.items():
            for key, value in params.items():
                if key == 'active': continue
                env_key = f"{prefix}{category.upper()}_{key.upper()}"
                if isinstance(value, list):
                    if isinstance(value[0], list):  # Nested ranges
                        for i, subrange in enumerate(value):
                            ENV.save(f"{env_key}_{i}_LO", subrange[0])
                            ENV.save(f"{env_key}_{i}_HI", subrange[1])
                    else:
                        ENV.save(f"{env_key}_LO", value[0])
                        ENV.save(f"{env_key}_HI", value[1])
                else:
                    ENV.save(env_key, value)

    def load_env_var(self, widget):
        """Load category parameters from environment."""
        prefix = 'WEED_DETECTOR_' if widget == 'detection' else 'CAMERA_CALIBRATION_'
        
        for category in self.categories:
            base_key = f"{prefix}{category.upper()}_"
            cat_params = self.categories[category]
            
            for param in cat_params:
                if param == 'active': continue
                env_key = f"{base_key}{param.upper()}"
                
                if isinstance(cat_params[param], list):
                    if any(isinstance(i, list) for i in cat_params[param]):
                        # Handle nested ranges
                        ranges = []
                        i = 0
                        while True:
                            lo = ENV.load(f"{env_key}_{i}_LO")
                            hi = ENV.load(f"{env_key}_{i}_HI")
                            if lo is None or hi is None: break
                            ranges.append([lo, hi])
                            i += 1
                        if ranges: self.categories[category][param] = ranges
                    else:
                        lo = ENV.load(f"{env_key}_LO")
                        hi = ENV.load(f"{env_key}_HI")
                        if lo is not None and hi is not None:
                            self.categories[category][param] = [lo, hi]
                else:
                    val = ENV.load(env_key)
                    if val is not None:
                        self.categories[category][param] = val

    def add_missing_params(self, widget):
        """Handle both legacy and category-based parameters."""
        if widget == 'calibration':
            for key, val in self.calibration_defaults.items():
                if key not in self.legacy_defaults:
                    self.legacy_defaults[key] = val
                    
        for category in self.categories:
            # Merge legacy parameters into category params
            for key in ['blur', 'morph', 'iterations', 'H', 'S', 'V']:
                if key not in self.categories[category]:
                    self.categories[category][key] = self.legacy_defaults[key]
            
            # Add size limits if missing
            if 'min_size' not in self.categories[category]:
                self.categories[category]['min_size'] = self.SIZE_LIMITS[category][0]
            if 'max_size' not in self.categories[category]:
                self.categories[category]['max_size'] = self.SIZE_LIMITS[category][1]

    def print_input(self):
        """Print parameters with category breakdown."""
        print('Processing Parameters:')
        print('-' * 25)
        for category in self.detection_order:
            if not self.categories[category]['active']: continue
            params = self.categories[category]
            print(f"\n{category.upper()} Detection:")
            print(f"Blur: {params['blur']}")
            print(f"Morph: {params['morph']}")
            print(f"Iterations: {params['iterations']}")
            print(f"Hue Range: {params['H']}")
            print(f"Saturation Range: {params['S']}")
            print(f"Value Range: {params['V']}")
            print(f"Size Range: {params['min_size']} - {params['max_size']} mm²")
            print(f"Adaptive S: {params['adaptive_s']}, Adaptive V: {params['adaptive_v']}")
            print(f"Edge Preservation: {params['edge_preservation']}")
        print('-' * 25)

    def visualize_parameters(self, base_image):
        """Create parameter visualization overlay"""
        overlay = np.zeros_like(base_image)
        
        # Blur visualization (cyan)
        blur_value = self.categories['crop']['blur']
        cv2.circle(overlay, (50, 50), blur_value*2, 
                  self.COLOR_SCHEME['blur'], -1)
        
        # Morph visualization (magenta)
        morph_value = self.categories['weed']['morph']
        cv2.rectangle(overlay, (150, 50), 
                    (150+morph_value*2, 50+morph_value*2),
                    self.COLOR_SCHEME['morph'], -1)
        
        # Iterations visualization (yellow)
        iterations = self.categories['grass']['iterations']
        for i in range(iterations):
            cv2.line(overlay, (250+i*20, 50), (250+i*20, 70),
                    self.COLOR_SCHEME['iterations'], 3)
                    
        return cv2.addWeighted(base_image, 0.7, overlay, 0.3, 0)

    def adapt_threshold(self, image, channel):
        """Auto-adjust based on image histogram"""
        if channel == 'saturation':
            channel_idx = 1
        elif channel == 'value':
            channel_idx = 2
        else:
            channel_idx = 0  # Hue channel
            
        hist = cv2.calcHist([image.hsv], [channel_idx], None, [256], [0, 256])
        # Implement histogram analysis to find optimal cutoff
        self.current_threshold = np.argmax(hist > np.percentile(hist, 85))
        return self.current_threshold

    def optimize_edge_filter(self, category):
        """Set edge filter parameters based on category complexity"""
        complexity = self.categories[category].get('edge_preservation', 'medium')
        self.sigma_s, self.sigma_r = self.EDGE_PRESERVATION[complexity]
        return self.sigma_s, self.sigma_r

    def detect_plants(self, image):
        """Detect plants in different categories"""
        # Category detection with color coding
        result = np.zeros_like(image.original)
        
        # Process each active category
        for category in self.detection_order:
            if not self.categories[category]['active']:
                continue
                
            # Process category and get mask
            mask = self.process_category(image, category)
            
            # Apply category color to result
            result[mask > 0] = self.DETECTION_COLORS[category]
        
        # Combine with original image
        blended = cv2.addWeighted(image.original, 0.6, result, 0.4, 0)
        
        # Add parameter visualization
        final_display = self.visualize_parameters(blended)
        
        return final_display

    def process_category(self, image, category):
        """Apply category-specific processing"""
        params = self.categories[category]
        
        # Create copy of the HSV image for processing
        hsv = image.hsv.copy()
        
        # Apply blur
        blurred = cv2.GaussianBlur(hsv, (params['blur'] * 2 + 1, params['blur'] * 2 + 1), 0)
        
        # Adaptive thresholding if enabled
        if params.get('adaptive_s', False):
            s_threshold = self.adapt_threshold(image, 'saturation')
            params['S'][0] = s_threshold
            
        if params.get('adaptive_v', False):
            v_threshold = self.adapt_threshold(image, 'value')
            params['V'][0] = v_threshold
        
        # Create mask based on HSV range
        mask = np.zeros(image.original.shape[:2], dtype=np.uint8)
        
        # Handle complex hue ranges (for red which wraps around)
        if isinstance(params['H'][0], list):
            for h_range in params['H']:
                h_mask = cv2.inRange(
                    blurred,
                    (h_range[0], params['S'][0], params['V'][0]),
                    (h_range[1], params['S'][1], params['V'][1])
                )
                mask = cv2.bitwise_or(mask, h_mask)
        else:
            mask = cv2.inRange(
                blurred,
                (params['H'][0], params['S'][0], params['V'][0]),
                (params['H'][1], params['S'][1], params['V'][1])
            )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(
            self.cv2_kt.get(self.kernel_type, cv2.MORPH_ELLIPSE),
            (params['morph'] * 2 + 1, params['morph'] * 2 + 1)
        )
        
        for _ in range(params['iterations']):
            morph_type = self.cv2_mt.get(self.morph_type, cv2.MORPH_CLOSE)
            if morph_type == 'erode':
                mask = cv2.erode(mask, kernel)
            elif morph_type == 'dilate':
                mask = cv2.dilate(mask, kernel)
            else:
                mask = cv2.morphologyEx(mask, morph_type, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Size filtering
        valid_contours = [
            c for c in contours 
            if params['min_size'] < cv2.contourArea(c) < params['max_size']
        ]
        
        # Create mask from filtered contours
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(filtered_mask, valid_contours, -1, 255, -1)
        
        # Edge-aware processing
        sigma_s, sigma_r = self.optimize_edge_filter(category)
        processed_mask = cv2.edgePreservingFilter(
            filtered_mask, flags=cv2.NORMCONV_FILTER,
            sigma_s=sigma_s, sigma_r=sigma_r
        )
        
        return processed_mask
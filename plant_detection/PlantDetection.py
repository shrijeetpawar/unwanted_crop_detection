
#!/usr/bin/env python
"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import sys
import os

import cv2
from plant_detection.P2C import Pixel2coord
from plant_detection.Image import Image
from plant_detection.Parameters import Parameters
from plant_detection.DB import DB
from plant_detection.Log import log
from plant_detection.hybrid_detection import run_hybrid_detection
from .EnhancedEpd import EnhancedEpd
from .yolo_opencv_detector import YoloCropWeedDetector
# from plant_detection.Capture import Capture


class PlantDetection(object):

    def __init__(self, **kwargs):
        """Read arguments (and change settings) and initialize modules."""
        # Default Data Inputs
        self.image = None
        self.plant_db = DB()

        self.detector = EnhancedEpd(model_type='yolov5', backend='opencv')  # or 'maskrcnn'
        self.plant_db.tmp_dir = None


        # Default Parameter Inputs
        self.params = Parameters()
        self.params.add_missing_params('detect')

        # Load keyword argument inputs
        self._data_inputs(kwargs)
        self._parameter_inputs(kwargs)
        self.args = kwargs

        # Set remaining arguments to defaults
        self._set_defaults()

        # Changes based on inputs
        if self.args['calibration_img'] is not None:
            #self.coordinates = True
            self.args['coordinates'] = True
        if self.args['GUI']:
            self.args['save'] = False
            self.args['text_output'] = False
        if self.args['app']:
            self.args['verbose'] = False
            self.args['from_env_var'] = True
            self.plant_db.app = True

        # Remaining initialization
        self.p2c = None
        #self.capture = Capture().capture
        self.final_marked_image = None
        self.plant_db.tmp_dir = None

    def _set_defaults(self):
        default_args = {
            # Default Data Inputs
            'image': None, 'calibration_img': None, 'known_plants': None,
            'app_image_id': None, 'calibration_data': None,
            # Default Program Options
            'coordinates': False, 'from_file': False, 'from_env_var': False,
            'clump_buster': False, 'GUI': False, 'app': False,
            # Default Output Options
            'debug': False, 'save': True,
            'text_output': True, 'verbose': True, 'print_all_json': False,
            'output_celeryscript_points': False,
            # Default Graphic Options
            'grey_out': False, 'draw_contours': True, 'circle_plants': True,
            # Default processing options
            'array': None,
            'blur': self.params.parameters['blur'],
            'morph': self.params.parameters['morph'],
            'iterations': self.params.parameters['iterations'],
            'HSV_min': [self.params.parameters['H'][0],
                        self.params.parameters['S'][0],
                        self.params.parameters['V'][0]],
            'HSV_max': [self.params.parameters['H'][1],
                        self.params.parameters['S'][1],
                        self.params.parameters['V'][1]],
        }
        for key, value in default_args.items():
            if key not in self.args:
                self.args[key] = value

    def _data_inputs(self, kwargs):
        """Load data inputs from keyword arguments."""
        for key in kwargs:
            if key == 'known_plants':
                self.plant_db.plants['known'] = kwargs[key]

    def _parameter_inputs(self, kwargs):
        """Load parameter inputs from keyword arguments."""
        for key in kwargs:
            if key == 'blur':
                self.params.parameters['blur'] = kwargs[key]
            if key == 'morph':
                self.params.parameters['morph'] = kwargs[key]
            if key == 'iterations':
                self.params.parameters['iterations'] = kwargs[key]
            if key == 'array':
                self.params.array = kwargs[key]
            if key == 'HSV_min':
                hsv_min = kwargs[key]
                self.params.parameters['H'][0] = hsv_min[0]
                self.params.parameters['S'][0] = hsv_min[1]
                self.params.parameters['V'][0] = hsv_min[2]
            if key == 'HSV_max':
                hsv_max = kwargs[key]
                self.params.parameters['H'][1] = hsv_max[0]
                self.params.parameters['S'][1] = hsv_max[1]
                self.params.parameters['V'][1] = hsv_max[2]

    def _calibration_input(self):  # provide inputs to calibration
        import os
        if self.args['app_image_id'] is not None:
            self.args['calibration_img'] = int(self.args['app_image_id'])
        #if self.args['calibration_img'] is None and self.args['coordinates']:
            # Calibration requested, but no image provided.
            # Take a calibration image.
            # self.args['calibration_img'] = self.capture()

        # Set calibration input parameters
        if self.args['from_env_var']:
            calibration_input = 'env_var'
        elif self.args['from_file']:  # try to load from file
            calibration_input = 'file'
        else:  # Use default calibration inputs
            calibration_input = None

        # Check if calibration image exists and is valid
        calib_img_path = self.args.get('calibration_img')
        if calib_img_path is None or not os.path.isfile(calib_img_path):
            # Use default calibration image if available
            default_calib_img = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'p2c_test_calibration.jpg')
            if os.path.isfile(default_calib_img):
                calib_img_path = default_calib_img
                self.args['calibration_img'] = calib_img_path
            else:
                raise FileNotFoundError("Calibration image not found. Please provide a valid calibration image.")

        # Call coordinate conversion module
        self.p2c = Pixel2coord(self.plant_db,
                               calibration_image=calib_img_path,
                               calibration_data=self.args['calibration_data'],
                               load_data_from=calibration_input)
        self.p2c.debug = self.args['debug']

        # Ensure image attribute is initialized for calibration
        if self.p2c.image is None:
            self.p2c.image = Image(self.params, self.plant_db)
            if calib_img_path is not None:
                self.p2c.image.load(calib_img_path)



    def calibrate(self):
        """Calibrate the camera for plant detection."""
        print("[DEBUG] Starting calibration process...")
        
        # Initialize calibration image path
        calib_img_path = self.args.get('calibration_img')
        if calib_img_path is None:
            calib_img_path = os.path.join(os.path.dirname(__file__), 
                                        "p2c_test_calibration.jpg")
            self.args['calibration_img'] = calib_img_path

        print(f"[DEBUG] Using calibration image: {calib_img_path}")
        
        # Verify image exists
        if not os.path.exists(calib_img_path):
            print(f"[ERROR] Calibration image not found at: {calib_img_path}")
            return False

        try:
            # Initialize calibration with debug mode
            self._calibration_input()
            self.p2c.debug = True
            self.p2c.image.debug = True
            self.p2c.image.calibration_debug = True
            
            # Save intermediate processing images
            self.p2c.image.save('calibration_original')
            
            # Process the image with current parameters
            print("\n[DEBUG] Image Processing Parameters:")
            print(f"Blur: {self.params.parameters['blur']}")
            print(f"Morph: {self.params.parameters['morph']}")
            print(f"Hue: {self.params.parameters['H']}")
            print(f"Saturation: {self.params.parameters['S']}")
            print(f"Value: {self.params.parameters['V']}")
            
            # Run initial processing and save each step
            self.p2c.image.initial_processing()
            self.p2c.image.save('calibration_blurred')
            self.p2c.image.save('calibration_masked')
            self.p2c.image.save('calibration_morphed')
            
            # Find objects with debug info
            print("\n[DEBUG] Searching for calibration objects...")
            self.p2c.image.find(calibration=True)
            
            if not hasattr(self.plant_db, 'calibration_pixel_locations') or \
            len(self.plant_db.calibration_pixel_locations) < 2:
                print("\n[DEBUG] Failed to detect calibration objects. Possible issues:")
                print("- Objects not contrasting enough with background")
                print("- HSV parameters may need adjustment")
                print("- Try increasing blur/morph values")
                print("\nCheck saved calibration_*.jpg images to see processing stages")
                return False
                
            # Perform calibration
            print("\n[DEBUG] Performing coordinate calibration...")
            exit_flag = self.p2c.calibration()
            
            if exit_flag:
                print("[ERROR] Calibration failed during processing")
                return False
                
            print("\n[DEBUG] Calibration successful!")
            print(f"Rotation angle: {self.p2c.calibration_params['total_rotation_angle']}")
            print(f"Coordinate scale: {self.p2c.calibration_params['coord_scale']}")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Calibration failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    # def calibrate(self):
    #     """Calibrate the camera for plant detection.

    #     Initialize the coordinate conversion module using a calibration image,
    #     perform calibration, and save calibration data.
    #     """
    #     self._calibration_input()  # initialize coordinate conversion module
    #     try:
    #         exit_flag = self.p2c.calibration()  # perform calibration
    #         if exit_flag:
    #             sys.exit(0)
    #     except Exception as e:
    #         print(f"[ERROR] Calibration failed: {str(e)}")
    #         print("[ERROR] Please verify the calibration image and parameters.")
    #         sys.exit(1)
    #     self._calibration_output()  # save calibration data

    def _calibration_output(self):  # save calibration data
        if self.args['save'] or self.args['debug']:
            self.p2c.image.images['current'] = self.p2c.image.images['marked']
            self.p2c.image.save('calibration_result')

        # Print verbose results
        if self.args['verbose'] and self.args['text_output']:
            if self.p2c.calibration_params['total_rotation_angle'] != 0:
                print(" Note: required rotation of "
                      "{:.2f} degrees executed.".format(
                          self.p2c.calibration_params['total_rotation_angle']))
            if self.args['debug']:
                # print number of objects detected
                self.plant_db.print_count(calibration=True)
                # print coordinate locations of calibration objects
                self.p2c.p2c(self.plant_db)
                self.plant_db.print_coordinates()
                print('')

        # Print condensed output if verbose output is not chosen
        if self.args['text_output'] and not self.args['verbose']:
            print("Calibration complete. (rotation:{}, scale:{})".format(
                self.p2c.calibration_params['total_rotation_angle'],
                self.p2c.calibration_params['coord_scale']))

        # Send calibration result log toast
        if self.args['app']:
            log('Camera calibration complete; setting pixel coordinate scale'
                ' to {} and camera rotation to {} degrees.'.format(
                    self.p2c.calibration_params['coord_scale'],
                    self.p2c.calibration_params['total_rotation_angle']),
                'success', 'Success', ['toast'], True)

        # Save calibration data
        if self.args['from_env_var']:
            # to environment variable
            self.p2c.save_calibration_data_to_env()
        elif self.args['from_file']:  # to file
            self.p2c.save_calibration_parameters()
        else:  # to Parameters() instance
            self.params.calibration_data = self.p2c.calibration_params

    def _detection_input(self):  # provide input to detect_plants
        # Load input parameters
        if self.args['from_file']:
            # Requested to load detection parameters from file
            try:
                self.params.load('detect')
            except IOError:
                print("Warning: Input parameter file load failed. "
                      "Using defaults.")
            self.plant_db.load_plants_from_file()
        if self.args['app']:
            self.plant_db.load_plants_from_web_app()
        if self.args['from_env_var']:
            # Requested to load detection parameters from json ENV variable
            self.params.load_env_var('detect')

        # Print input parameters and filename of image to process
        if self.args['verbose'] and self.args['text_output']:
            self.params.print_input()
            print("\nProcessing image: {}".format(self.args['image']))

    def _detection_image(self):  # get image to process
        self.image = Image(self.params, self.plant_db)
        # Get image to process
        try:  # check for API image ID
            image_id = self.args['app_image_id']
        except KeyError:
            image_id = None
        if image_id is not None:  # download image
            try:
                self.image.download(image_id)
            except IOError:
                print("Image download failed for image ID {}.".format(
                    str(image_id)))
                sys.exit(0)
        #elif self.args['image'] is None:  # No image provided. Capture one.
            #self.image.capture()
            # if self.args['debug']:
            #     self.image.save('photo')
        else:  # Image provided. Load it.
            filename = self.args['image']
            self.image.load(filename)
        self.image.debug = self.args['debug']

    def _coordinate_conversion(self):  # determine detected object coordinates
        # Load calibration data
        load_data_from = None
        calibration_data = None
        if self.args['from_env_var']:
            load_data_from = 'env_var'
        elif self.args['from_file']:
            load_data_from = 'file'
        else:  # use data saved in self.params
            calibration_data = self.params.calibration_data
        # Initialize coordinate conversion module
        self.p2c = Pixel2coord(
            self.plant_db,
            load_data_from=load_data_from, calibration_data=calibration_data)
        self.p2c.debug = self.args['debug']
        # Check for coordinate conversion calibration results
        present = {'coord_scale': False,
                   'camera_z': False,
                   'center_pixel_location': False,
                   'total_rotation_angle': False}
        try:
            for key in present:
                present[key] = self.p2c.calibration_params[key]
        except KeyError:
            log("ERROR: Coordinate conversion calibration values "
                "not found. Run calibration first.",
                message_type='error', title='plant-detection')
            sys.exit(0)
        # Validate coordinate conversion calibration data for image
        calibration_data_valid = self.p2c.validate_calibration_data(
            self.image.images['current'])
        if not calibration_data_valid:
            log("ERROR: Coordinate conversion calibration values "
                "invalid for provided image.",
                message_type='error', title='plant-detection')
            sys.exit(0)
        # Determine object coordinates
        self.image.coordinates(self.p2c,
                               draw_contours=self.args['draw_contours'])
        # Organize objects into plants and weeds
        self.plant_db.identify(self.params.parameters)
        if self.plant_db.plants['safe_remove']:
            self.image.safe_remove(self.p2c)

    def _coordinate_conversion_output(self):  # output detected object data
        # Print and output results
        if self.args['text_output']:
            self.plant_db.print_count()  # print number of objects detected
        if self.args['verbose'] and self.args['text_output']:
            self.plant_db.print_identified()  # print organized plant data
        if self.args['output_celeryscript_points']:
            self.plant_db.output_celery_script()  # print points JSON to stdout
        if self.args['app']:
            save_detected_plants = self.params.parameters['save_detected_plants']
            # add detected weeds and points to FarmBot Web App
            self.plant_db.upload_plants(save_detected_plants)
        if self.args['debug']:
            self.image.save_annotated('contours')
            self.image.images['current'] = self.image.images['marked']
            self.image.save_annotated('coordinates_found')
        if self.args['circle_plants']:
            self.image.label(self.p2c)  # mark objects with colored circles
        self.image.grid(self.p2c)  # add coordinate grid and features

    
    # In PlantDetection.py, modify the detect_plants method:

    # def detect_plants(self):
    #     # Auto calibrate if calibration data is missing
    #     if not self.params.calibration_data:
    #         print("[INFO] Calibration data missing. Running calibration...")
    #         self.calibrate()

    #     try:
    #         # 🧠 Step 1: Run Hybrid Detection (Roboflow + Local Mask R-CNN)
    #         img, detections = run_hybrid_detection(self.args['image'])

    #         # Create Image object and load original image
    #         self.image = Image(self.params, self.plant_db)
    #         self.image.load(self.args['image'])

    #         # Save hybrid result
    #         self.image.images['marked'] = img
    #         self.final_marked_image = img

    #         print(f"[INFO] Hybrid detection completed with {len(detections)} detections.")

    #         # Coordinate conversion and output
    #         try:
    #             self._coordinate_conversion()
    #             self._coordinate_conversion_output()
    #         except SystemExit:
    #             print("[ERROR] Coordinate conversion calibration values not found. Please run calibration first.")

    #     except Exception as e:
    #         print(f"[WARNING] Hybrid detection failed: {str(e)}")
    #         print("[INFO] Falling back to YOLO OpenCV detection...")

    #         try:
    #             # 🧪 Step 2: YOLO fallback
    #             weights_path = os.path.join(os.path.dirname(__file__),
    #                 "Crop_and_weed_detection-master/performing_detection/data/weights/crop_weed_detection.weights")
    #             config_path = os.path.join(os.path.dirname(__file__),
    #                 "Crop_and_weed_detection-master/performing_detection/data/cfg/crop_weed.cfg")
    #             names_path = os.path.join(os.path.dirname(__file__),
    #                 "Crop_and_weed_detection-master/performing_detection/data/names/obj.names")

    #             detector = YoloCropWeedDetector(
    #                 weights_path=weights_path,
    #                 config_path=config_path,
    #                 names_path=names_path,
    #                 conf_threshold=0.4
    #             )

    #             result = detector.detect(self.args['image'])

    #             self.image = Image(self.params, self.plant_db)
    #             self.image.load(self.args['image'])
    #             self.image.images['marked'] = result
    #             self.final_marked_image = result

    #             output_path = os.path.join(os.path.dirname(self.args['image']), "detection_result.jpg")
    #             cv2.imwrite(output_path, result)

    #             print("[INFO] YOLO detection completed.")

    #             # Coordinate conversion and output
    #             try:
    #                 self._coordinate_conversion()
    #                 self._coordinate_conversion_output()
    #             except SystemExit:
    #                 print("[ERROR] Coordinate conversion calibration values not found. Please run calibration first.")

    #         except Exception as fallback_error:
    #             print(f"[ERROR] YOLO detection also failed: {str(fallback_error)}")
    #             print("[INFO] Running traditional OpenCV detection as last resort...")

    #             self._detection_input()
    #             self._detection_image()
    #             self.image.initial_processing()

    #             if self.args['clump_buster']:
    #                 self.image.clump_buster()
    #             if self.args['grey_out']:
    #                 self.image.grey()

    #             self.image.find(draw_contours=self.args['draw_contours'])

    #             if self.args['circle_plants']:
    #                 self.image.label()

    #             self.final_marked_image = self.image.images['marked']

    #             # Morphological post-process (optional)
    #             if self.final_marked_image is not None:
    #                 gray = cv2.cvtColor(self.final_marked_image, cv2.COLOR_BGR2GRAY)
    #                 _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #                 kernel = np.ones((3, 3), np.uint8)
    #                 opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    #                 self.image.images['marked'] = cv2.bitwise_and(self.final_marked_image, self.final_marked_image, mask=opening)

    #             # Coordinate conversion and output
    #             try:
    #                 self._coordinate_conversion()
    #                 self._coordinate_conversion_output()
    #             except SystemExit:
    #                 print("[ERROR] Coordinate conversion calibration values not found. Please run calibration first.")

    #     # Final stage: save and show output
    #     self._show_detection_output()
    #     self._save_detection_output()



    def detect_plants(self):
        """Detect plants with three-color classification and fallback handling."""
        print("[INFO] Starting enhanced plant detection process...")
        
        # Step 1: Calibration check
        if not self.params.calibration_data:
            print("[INFO] No calibration data found, running calibration...")
            if not self.calibrate():
                print("[ERROR] Calibration failed, cannot proceed with detection")
                return False

        try:
            # Step 2: Load and verify image
            self._detection_image()
            if self.image.images['original'] is None:
                print("[ERROR] Failed to load detection image")
                return False

            # Step 3: Attempt three-color detection
            print("[INFO] Attempting three-color classification...")
            base_img = cv2.cvtColor(self.image.images['original'], cv2.COLOR_BGR2HSV)
            final_output = np.zeros_like(self.image.images['original'])
            
            # Color-based detection for each category
            detection_success = False
            for category in self.params.detection_order:
                params = self.params.parameters[category]
                
                try:
                    # Create mask with category-specific parameters
                    blurred = cv2.GaussianBlur(base_img, (params['blur'], params['blur']), 0)
                    
                    if category == 'weed':
                        # Dual range handling for red colors
                        mask1 = cv2.inRange(blurred, 
                            (params['H'][0][0], params['S'][0], params['V'][0]),
                            (params['H'][0][1], params['S'][1], params['V'][1]))
                        mask2 = cv2.inRange(blurred, 
                            (params['H'][1][0], params['S'][0], params['V'][0]),
                            (params['H'][1][1], params['S'][1], params['V'][1]))
                        mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        mask = cv2.inRange(blurred, 
                            (params['H'][0], params['S'][0], params['V'][0]),
                            (params['H'][1], params['S'][1], params['V'][1]))

                    # Morphological processing
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                (params['morph'], params['morph']))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                iterations=params['iterations'])
                    
                    # Apply color coding with priority
                    final_output[mask > 0] = params['color']
                    detection_success = True
                    
                except Exception as e:
                    print(f"[WARNING] {category} detection failed: {str(e)}")
                    detection_success = False
                    break

            if detection_success:
                # Post-processing
                final_output = cv2.medianBlur(final_output, 5)
                self.final_marked_image = cv2.addWeighted(
                    self.image.images['original'], 0.7, 
                    final_output, 0.3, 0)
                
                # Coordinate processing
                self._process_coordinates()
                self._save_detection_output()
                return True

        except Exception as e:
            print(f"[WARNING] Three-color detection failed: {str(e)}")

        # Fallback to hybrid detection
        try:
            print("[INFO] Falling back to hybrid detection...")
            img, detections = run_hybrid_detection(self.args['image'])
            self.image.images['marked'] = img
            self.final_marked_image = img
            self._process_coordinates()
            return True
            
        except Exception as e:
            print(f"[WARNING] Hybrid detection failed: {str(e)}")
            print("[INFO] Falling back to traditional detection...")

            # Traditional detection fallback
            try:
                self.image.initial_processing()
                if self.args['clump_buster']:
                    self.image.clump_buster()
                if self.args['grey_out']:
                    self.image.grey()
                self.image.find(draw_contours=self.args['draw_contours'])
                self._process_coordinates()
                return True
                
            except Exception as e:
                print(f"[ERROR] All detection methods failed: {str(e)}")
                return False

        # Final output
        self._show_detection_output()
        self._save_detection_output()
        return True



    # def detect_plants(self):
    #     """Detect plants with improved error handling."""
    #     print("[INFO] Starting plant detection process...")

    #     # Load and prepare base image
    #     self._detection_image()
    #     base_img = cv2.cvtColor(self.image.images['original'], cv2.COLOR_BGR2HSV)
    #     final_output = np.zeros_like(self.image.images['original'])
        
    #     # Create detection layer for each category
    #     for category in self.params.detection_order:
    #         params = self.params.parameters[category]
            
    #         # Create mask
    #         blurred = cv2.GaussianBlur(base_img, (params['blur'], params['blur']), 0)
            
    #         # Handle dual HSV ranges for weeds
    #         if category == 'weed':
    #             mask1 = cv2.inRange(blurred, 
    #                 (params['H'][0][0], params['S'][0], params['V'][0]),
    #                 (params['H'][0][1], params['S'][1], params['V'][1]))
    #             mask2 = cv2.inRange(blurred, 
    #                 (params['H'][1][0], params['S'][0], params['V'][0]),
    #                 (params['H'][1][1], params['S'][1], params['V'][1]))
    #             mask = cv2.bitwise_or(mask1, mask2)
    #         else:
    #             mask = cv2.inRange(blurred, 
    #                 (params['H'][0], params['S'][0], params['V'][0]),
    #                 (params['H'][1], params['S'][1], params['V'][1]))
            
    #         # Morphological operations
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
    #                     (params['morph'], params['morph']))
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
    #                     iterations=params['iterations'])
            
    #         # Apply to final output with priority
    #         final_output[mask > 0] = params['color']
        
    #     # Post-processing
    #     final_output = cv2.medianBlur(final_output, 5)
    #     self.final_marked_image = cv2.addWeighted(
    #         self.image.images['original'], 0.7, 
    #         final_output, 0.3, 0)
        
    #     # Save and show results
    #     self._save_detection_output()
    #     self._show_detection_output()
    #     return True

        
    #     # Step 1: Ensure we have calibration data
    #     if not self.params.calibration_data:
    #         print("[INFO] No calibration data found, running calibration...")
    #         if not self.calibrate():
    #             print("[ERROR] Calibration failed, cannot proceed with detection")
    #             return False
        
    #     # Step 2: Load and verify the detection image
    #     try:
    #         self._detection_image()
    #         if self.image.images['original'] is None:
    #             print("[ERROR] Failed to load detection image")
    #             return False
    #     except Exception as e:
    #         print(f"[ERROR] Image loading failed: {str(e)}")
    #         return False
        
    #     # Step 3: Try hybrid detection first
    #     try:
    #         print("[INFO] Attempting hybrid detection...")
    #         img, detections = run_hybrid_detection(self.args['image'])
    #         self.image.images['marked'] = img
    #         self.final_marked_image = img
    #         print(f"[INFO] Hybrid detection found {len(detections)} objects")
            
    #         # Process coordinates
    #         self._process_coordinates()
            
    #     except Exception as e:
    #         print(f"[WARNING] Hybrid detection failed: {str(e)}")
    #         print("[INFO] Falling back to traditional detection...")
            
    #         # Traditional detection fallback
    #         try:
    #             self.image.initial_processing()
                
    #             if self.args['clump_buster']:
    #                 self.image.clump_buster()
                    
    #             if self.args['grey_out']:
    #                 self.image.grey()
                    
    #             self.image.find(draw_contours=self.args['draw_contours'])
                
    #             # Process coordinates
    #             self._process_coordinates()
                
    #         except Exception as e:
    #             print(f"[ERROR] Traditional detection failed: {str(e)}")
    #             return False
        
    #     # Final output
    #     self._show_detection_output()
    #     self._save_detection_output()
    #     return True

    def _process_coordinates(self):
        """Handle coordinate conversion and output."""
        try:
            print("[INFO] Processing coordinates...")
            self._coordinate_conversion()
            self._coordinate_conversion_output()
            
            # Print coordinates in (x,y) format
            if hasattr(self.plant_db, 'coordinate_locations') and self.plant_db.coordinate_locations:
                print("\nDetected Plant Coordinates:")
                for i, coord in enumerate(self.plant_db.coordinate_locations):
                    x, y, radius = coord
                    print(f"Plant {i+1}: ({x:.1f}, {y:.1f}) - Radius: {radius:.1f}mm")
            else:
                print("[WARNING] No coordinate data available")
                
        except Exception as e:
            print(f"[ERROR] Coordinate processing failed: {str(e)}")


    def _show_detection_output(self):  # show detect_plants output
        # Print raw JSON to STDOUT
        if self.args['print_all_json']:
            print("\nJSON:")
            print(self.params.parameters)
            print(self.plant_db.plants)
            if self.p2c is not None:
                print(self.p2c.calibration_params)

        # Print condensed inputs if verbose output is not chosen
        if self.args['text_output'] and not self.args['verbose']:
            print('{}: {}'.format('known plants input',
                                  self.plant_db.plants['known']))
            print('{}: {}'.format('parameters input',
                                  self.params.parameters))
            print('{}: {}'.format('coordinates input',
                                  self.plant_db.coordinates))

    def _save_detection_output(self):  # save detect_plants output
        import platform
        import subprocess
        # Final marked image
        if self.args['save'] or self.args['debug']:
            self.image.save('marked')
            # Open the saved image automatically on Windows
            if platform.system() == 'Windows':
                try:
                    saved_path = self.image.get_save_path('marked')
                    if saved_path:
                        import os
                        os.startfile(saved_path)
                except Exception as e:
                    print(f"[WARNING] Could not open saved image automatically: {e}")
        elif self.args['GUI']:
            self.final_marked_image = self.image.images['marked']

        # Save input parameters
        if self.args['from_env_var']:
            # to environment variable
            self.params.save_to_env_var('detect')
        elif self.args['save']:
            # to file
            self.params.save()
        elif self.args['GUI']:
            # to file for GUI
            self.params.save()

        # Save plants
        if self.args['save']:
            self.plant_db.save_plants()
    


if __name__ == "__main__":
    import time
    DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
    IMG = 'e:/notes/Project/plant-detection/plant_detection/p2c_test_objects.jpg'
    CALIBRATION_IMG = DIR + "p2c_test_calibration.jpg"
    if len(sys.argv) == 1:
        PD = PlantDetection(
            image=IMG,
            blur=15, morph=6, iterations=4,
            calibration_img=CALIBRATION_IMG,
            known_plants=[{'x': 200, 'y': 600, 'radius': 100},
                          {'x': 900, 'y': 200, 'radius': 120}])
        print("[INFO] Starting calibration...")
        PD.calibrate()  # use calibration img to get coordinate conversion data
        print("[INFO] Calibration complete.")
        print("[INFO] Starting plant detection...")
        PD.detect_plants()  # detect coordinates and sizes of weeds and plants
        print("[INFO] Plant detection complete.")
        # Wait a bit to allow user to see any image window before script exits
        time.sleep(5)
    else:  # command line argument(s)
        if sys.argv[1] == '--GUI':
            from plant_detection.GUI import PlantDetectionGUI
            if len(sys.argv) == 3:  # image filename provided
                GUI = PlantDetectionGUI(image_filename=sys.argv[2],plant_detection=PlantDetection)

            else:  # Use `soil_image.jpg`
                GUI = PlantDetectionGUI(image_filename=IMG,plant_detection=PlantDetection)

            GUI.run()
        elif sys.argv[1] == '--cGUI':
            from plant_detection.GUI import CalibrationGUI
            if len(sys.argv) == 3:  # calibration image filename provided
                calibration_gui = CalibrationGUI(
                    cimage_filename=sys.argv[2],
                    image_filename=IMG,
                    plant_detection=PlantDetection)
            elif len(sys.argv) == 4:  # both image filenames provided
                calibration_gui = CalibrationGUI(
                    cimage_filename=sys.argv[2],
                    image_filename=sys.argv[3],
                    plant_detection=PlantDetection)
            else:  # Use `soil_image.jpg`
                calibration_gui = CalibrationGUI(
                    image_filename=IMG,
                    plant_detection=PlantDetection)
            calibration_gui.run()
        else:  # image filename provided
            IMG = sys.argv[1]
            PD = PlantDetection(
                image=IMG, from_file=True, debug=True)
            PD.detect_plants()

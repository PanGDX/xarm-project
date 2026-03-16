import os
import cv2
import numpy as np
import time
from scipy.spatial import distance
import sys
import matplotlib.pyplot as plt
from xarm.wrapper import XArmAPI
import time

class PathPlanner:
    def __init__(self, image_path, canvas_width_mm=200):
        self.image_path = image_path
        self.target_width = canvas_width_mm
        self.scale_factor = 1.0
        self.origin_offset = (0, 0)
        self.ordered_paths = []

    def display_step(self, img, title="Debug Step"):
        """Helper to visualize image processing steps using Matplotlib"""
        plt.figure(figsize=(10, 8))
        plt.title(title)
        # Check if image is grayscale or color
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show() # Code pauses here until window is closed

    def process_portrait(self, method="canny"):
        """
        Optimized for Faces and Fine Details with Visual Debugging.
        """
        # 1. Load Image
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not found")

        # 2. Resize maintaining aspect ratio
        h, w = img.shape
        new_w = 800
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)
        img = cv2.resize(img, (new_w, new_h))
        
        # --- DEBUG SHOW ---
        self.display_step(img, "1. Resized Image")

        # 3. Pre-processing: CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img = clahe.apply(img)
        # 4. Noise Reduction
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # # --- DEBUG SHOW ---
        # self.display_step(img, "3. Gaussian Blur (Noise Reduction)")
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)
        # --- DEBUG SHOW ---
        self.display_step(img, "3. Bilateral Filter (Smoothed Skin)")



        binary_output = None

        if method == "canny":
            # --- METHOD A: Canny Edge Detection ---
            v = np.median(img)
            lower_threshold = 20   # Increase this if you still have too much noise
            upper_threshold = 60  # Increase this if you still have too much noise
            
            edges = cv2.Canny(img, lower_threshold, upper_threshold)
#             If the nose/mouth are STILL missing: Lower the upper_threshold to 50 or 40.

# If the messy skin pores/shirt texture come back: Raise the upper_threshold back up slightly (e.g., 70 or 80), or increase the Bilateral Filter sigmaColor to 50.

# If the lines of the mouth are broken and dotted: Lower the lower_threshold to 10. This acts like a magnet, helping loose lines connect to stronger lines.
            # --- DEBUG SHOW ---
            self.display_step(edges, "4a. Raw Canny Edges (1px wide)")
            
            # Dilate to connect broken lines
            # kernel = np.ones((2,2), np.uint8)
            # dilated = cv2.dilate(edges, kernel, iterations=1)
            # binary_output = dilated

            dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1) # Connect gaps first
            binary_output = cv2.ximgproc.thinning(dilated)
            # --- DEBUG SHOW ---
            self.display_step(binary_output, "4b. Dilated (Thickened Lines)")
            
            

        elif method == "sketch":
            # --- METHOD B: Adaptive Thresholding ---
            binary_output = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 3
            )
            
            # --- DEBUG SHOW ---
            self.display_step(binary_output, "4. Adaptive Threshold Sketch")

        # 5. Extract Contours
        contours, hierarchy = cv2.findContours(
            binary_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 6. Filter Noise and Approximate
        min_path_length = 15 
        paths = []
        
        # Visualizing contours requires drawing them on a blank canvas
        debug_contours_canvas = np.zeros_like(img)
        
        for cnt in contours:
            if cv2.arcLength(cnt, False) > min_path_length:
                # Approximate
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                paths.append(approx.reshape(-1, 2))
                
                # Draw for debug
                cv2.drawContours(debug_contours_canvas, [approx], -1, 255, 1)

        # --- DEBUG SHOW ---
        self.display_step(debug_contours_canvas, "5. Final Contours (Robot Path)")

        self.scale_factor = self.target_width / new_w
        print(f"Details extracted. Found {len(paths)} paths.")
        return paths

    def optimize_paths(self, paths):
        """
        Greedy Algorithm (Nearest Neighbor) to minimize pen lifts.
        """
        if not paths:
            return []

        # Start at 0,0
        current_pos = np.array([0, 0])
        ordered = []
        remaining = paths.copy()

        # Pre-convert all paths to float arrays to avoid type errors during distance calc
        remaining = [p.astype(float) for p in remaining]

        print("Optimizing path order (this calculates travel distance)...")
        
        while remaining:
            # Get start and end points of all remaining paths
            starts = np.array([p[0] for p in remaining])
            ends = np.array([p[-1] for p in remaining])

            # Calculate distances from current position
            dist_to_starts = distance.cdist([current_pos], starts)[0]
            dist_to_ends = distance.cdist([current_pos], ends)[0]

            min_start_idx = np.argmin(dist_to_starts)
            min_end_idx = np.argmin(dist_to_ends)

            min_start_dist = dist_to_starts[min_start_idx]
            min_end_dist = dist_to_ends[min_end_idx]

            if min_start_dist < min_end_dist:
                best_idx = min_start_idx
                path_to_add = remaining.pop(best_idx)
            else:
                best_idx = min_end_idx
                path_to_add = remaining.pop(best_idx)
                path_to_add = path_to_add[::-1] # Reverse path

            ordered.append(path_to_add)
            current_pos = path_to_add[-1]

        self.ordered_paths = ordered
        return ordered

    def stream_data(self):
        for path in self.ordered_paths:
            # Move to start (Pen Up)
            start = path[0] * self.scale_factor
            yield (0, start[0], start[1])

            # Draw line (Pen Down)
            for point in path:
                pt = point * self.scale_factor
                yield (1, pt[0], pt[1])

class XArmArtist:
    def __init__(self, ip_address, speed=100, acceleration=2000, z_draw=0, z_travel=20, grip_width = 270):
        self.arm = XArmAPI(ip_address)
        self.z_draw = z_draw
        self.z_travel = z_travel
        self.speed = speed
        self.acc = acceleration
        self.grip_width = grip_width

    def connect(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_mode(0)
        time.sleep(1)
        print("Robot Connected.")

    def go_home(self):
        self.arm.move_gohome(wait=True)

    def put_pen_in(self):
        # Rotates wrist to side to make inserting pen easier
        self.arm.set_servo_angle(servo_id=5, angle=90, is_radian=False, wait=True)
        self.arm.set_gripper_position(self.grip_width+50, wait=True)
        print(">>> PLEASE INSERT PEN NOW. You have 10 seconds. <<<")
        time.sleep(10)
        self.arm.set_gripper_position(self.grip_width-30, wait=True) # Grip tight
        time.sleep(1)

    def draw(self, data_stream, origin_offset_x, origin_offset_y):
        self.go_home()
        print("Starting drawing...")

        current_state = 0  # 0 = Up, 1 = Down
        
        # Batch optimization could go here, but point-by-point is safer for beginners
        for cmd_type, x, y in data_stream:
            target_x = origin_offset_x + x
            target_y = origin_offset_y + y

            if cmd_type == 0:  # TRAVEL (Pen Up)
                if current_state == 1:
                    self.arm.set_position(z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 0
                self.arm.set_position(x=target_x, y=target_y, z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)

            elif cmd_type == 1:  # DRAW (Pen Down)
                if current_state == 0:
                    self.arm.set_position(z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 1
                # wait=False allows continuous motion (smoother curves)
                self.arm.set_position(x=target_x, y=target_y, z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=False)

        # Lift at end
        self.arm.set_position(z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
        self.go_home()
        self.arm.disconnect()
        print("Drawing complete.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_FILE = "./test_img/myself_rmv.jpg" 
    ROBOT_IP = "192.168.1.227"
    SIMULATION_MODE = True  # Set False to draw on real robot

    # PROCESS MODE: 'canny' (best for outlines/faces) or 'sketch' (messy shading)
    DRAWING_STYLE = "canny" 

    # ROBOT OFFSET
    ROBOT_ORIGIN_X = 250
    ROBOT_ORIGIN_Y = -100
    DRAW_WIDTH_MM = 150
    
    # CALIBRATION
    GRIPPER_DEPTH = 74.4
    PEN_LENGTH = 127.4 + 1.0
    PEN_DOWN_Z = PEN_LENGTH - GRIPPER_DEPTH
    PEN_UP_Z = PEN_DOWN_Z + 15

    planner = PathPlanner(IMAGE_FILE, canvas_width_mm=DRAW_WIDTH_MM)
    
    try:
        # USE THE NEW PORTRAIT METHOD
        raw_paths = planner.process_portrait(method=DRAWING_STYLE)
        planner.optimize_paths(raw_paths)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    stream = planner.stream_data()

    if SIMULATION_MODE:
        print("Running Simulation...")
        ink_x, ink_y = [], []
        travel_x, travel_y = [], []
        last_x, last_y = 0, 0
        
        for cmd, x, y in stream:
            if cmd == 0:
                travel_x.extend([last_x, x, np.nan])
                travel_y.extend([last_y, y, np.nan])
                ink_x.extend([np.nan])
                ink_y.extend([np.nan])
            elif cmd == 1:
                ink_x.extend([x])
                ink_y.extend([y])
            last_x, last_y = x, y

        plt.figure(figsize=(10, 10))
        plt.title(f"Simulation: {DRAWING_STYLE.upper()} Mode")
        plt.plot(travel_x, travel_y, 'r:', linewidth=0.3, alpha=0.3)
        plt.plot(ink_x, ink_y, 'k-', linewidth=0.8) # Black ink
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.show()

    else:
        print("Running on Real xArm...")
        import time 

        start = time.time()
        bot = XArmArtist(ROBOT_IP, speed=300, acceleration=1000, 
                         z_draw=PEN_DOWN_Z, z_travel=PEN_UP_Z, grip_width=270)
        bot.connect()
        bot.put_pen_in()
        bot.draw(stream, ROBOT_ORIGIN_X, ROBOT_ORIGIN_Y)
        print("Time Taken:")
        print(time.time() - start)

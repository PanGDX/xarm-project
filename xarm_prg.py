import os
import cv2
import numpy as np
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt
from xarm.wrapper import XArmAPI
from datetime import datetime
from rembg import remove
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

def select_file_via_explorer():
    """
    Opens a standard Windows file explorer dialog to select an image file.
    Works perfectly for USB drives or phones connected via USB (MTP).
    """
    # Create a dummy Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Force the window to appear on top
    root.attributes('-topmost', True)

    print("Opening file explorer...")
    file_path = filedialog.askopenfilename(
        title="Select an Image from USB/Phone",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
            ("All Files", "*.*")
        ]
    )
    
    # Destroy the dummy window after selection
    root.destroy()
    
    return file_path

class PathPlanner:
    def __init__(self, image_path, canvas_width_mm=400):
        self.image_path = image_path
        self.target_width = canvas_width_mm
        self.scale_factor = 1.0
        self.origin_offset = (0, 0)
        self.ordered_paths =[]

    def display_step(self, img, title="Debug Step", save=False):
        """Helper to visualize image processing steps using Matplotlib and optionally save them."""
        # --- NEW SAVING LOGIC ---
        if save:
            # Create an 'output' folder if it doesn't exist
            os.makedirs("output", exist_ok=True)

            # Strip special characters from the title to make a safe filename
            safe_title = "".join([c for c in title if c.isalnum() or c == ' ']).rstrip()
            
            image_file_name = Path(self.image_path).name
            
            now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            filename = f"output/{image_file_name} {safe_title.replace(' ', '_').lower() + ' ' + now}.png"

            # Save the raw image array cleanly using OpenCV
            cv2.imwrite(filename, img)
            print(f"[*] Saved debug image: {filename}")

        # --- EXISTING DISPLAY LOGIC ---
        plt.figure(figsize=(10, 8))
        plt.title(title)
        # Check if image is grayscale or color
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()  # Code pauses here until window is closed

    def process_portrait(self):
        """
        Optimized for Faces and Fine Details with Visual Debugging.
        """
        # 1. Load Image
        img_color = cv2.imread(self.image_path)
        if img_color is None:
            raise ValueError("Image not found")

        # 2. Remove Background
        # rembg takes a numpy array and returns a numpy array (BGRA format)
        img_nobg = remove(img_color)

        img = cv2.cvtColor(img_nobg, cv2.COLOR_BGRA2GRAY)

        # 2. Resize maintaining aspect ratio
        h, w = img.shape
        new_w = 800
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)
        img = cv2.resize(img, (new_w, new_h))

        # --- DEBUG SHOW ---
        self.display_step(img, "1. Resized Image, removed background", save=True)

        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)
        # --- DEBUG SHOW ---
        self.display_step(img, "3. Bilateral Filter (Smoothed Skin)")

        binary_output = None

        # --- METHOD A: Canny Edge Detection ---
        lower_threshold = 20   
        upper_threshold = 60  

        edges = cv2.Canny(img, lower_threshold, upper_threshold)

        # --- DEBUG SHOW ---
        self.display_step(
            edges, "4a. Raw Canny Edges (1px wide)", save=True)

        dilated = cv2.dilate(edges, np.ones(
            (3, 3), np.uint8), iterations=1)  # Connect gaps first
        binary_output = cv2.ximgproc.thinning(dilated)
        # --- DEBUG SHOW ---
        self.display_step(
            binary_output, "4b. Dilated (Thickened Lines)", save=True)


        # =====================================================================
        # NEW METHOD: Connected Component Analysis + Depth First Search
        # =====================================================================
        
        # 1. Get 8-connectivity components and their stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_output, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        
        # 2. Sort components by area (biggest to smallest), ignoring background (label 0)
        label_area_pairs = [(label, areas[label]) for label in range(1, num_labels)]
        label_area_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Group pixels by label for fast access
        y_coords, x_coords = np.nonzero(labels)
        label_vals = labels[y_coords, x_coords]
        
        component_pixels_dict = defaultdict(set)
        for y, x, lbl in zip(y_coords, x_coords, label_vals):
            component_pixels_dict[lbl].add((y, x))

        min_path_length = 15 # Reject components with less than 15 pixels
        paths =[]
        debug_contours_canvas = np.zeros_like(img)

        print(f"Found {num_labels - 1} connected components. Running DFS to generate paths...")

        # 4. Iterate over sorted components
        for label, area in label_area_pairs:
            if area < min_path_length:
                continue
                
            component_pixels = component_pixels_dict[label]
            
            # Find a good starting node (an endpoint with exactly 1 neighbor if possible)
            start_node = None
            for node in component_pixels:
                r, c = node
                neighbors =[(r-1, c-1), (r-1, c), (r-1, c+1), 
                             (r, c-1),             (r, c+1), 
                             (r+1, c-1), (r+1, c), (r+1, c+1)]
                count = sum(1 for n in neighbors if n in component_pixels)
                if count == 1:
                    start_node = node
                    break
            
            if start_node is None:
                # No endpoint found (e.g. a closed loop), just pick any node
                start_node = next(iter(component_pixels))
                
            # Iterative DFS with Backtracking to generate a single continuous path
            stack = [start_node]
            visited = {start_node}
            path =[]
            
            while stack:
                curr = stack[-1]
                # Add to path if it's the first node, or if we moved to a new node/backtracked
                if not path or path[-1] != curr:
                    path.append(curr)
                    
                r, c = curr
                neighbors =[(r-1, c-1), (r-1, c), (r-1, c+1), 
                             (r, c-1),             (r, c+1), 
                             (r+1, c-1), (r+1, c), (r+1, c+1)]
                             
                # Find valid unvisited neighbors
                unvisited_neighbors =[n for n in neighbors if n in component_pixels and n not in visited]
                
                if unvisited_neighbors:
                    next_node = unvisited_neighbors[0] # Pick the first available
                    visited.add(next_node)
                    stack.append(next_node)
                else:
                    stack.pop() # Dead end, remove from stack
                    if stack:
                        # Emulate physical backtracking (draw back over the line)
                        path.append(stack[-1]) 
                        
            # 5. Convert path to (x, y) coordinates
            path_xy = np.array([(c, r) for r, c in path], dtype=np.float32)
            
            # 6. Approximate path to reduce unnecessary points (G-code optimization)
            epsilon = 1.0
            approx = cv2.approxPolyDP(path_xy, epsilon, closed=False)
            
            if len(approx) > 1:
                final_path = approx.reshape(-1, 2)
                paths.append(final_path)
                cv2.polylines(debug_contours_canvas, [np.int32(final_path)], isClosed=False, color=255, thickness=1)

        # --- DEBUG SHOW ---
        self.display_step(debug_contours_canvas,
                          "5. Final DFS Paths (Robot Path)", save=True)

        self.scale_factor = self.target_width / new_w
        print(f"Details extracted. Found {len(paths)} paths.")
        
        # Logging
        image_file_name = Path(self.image_path).name
        os.makedirs("output", exist_ok=True)
        filename = "log.txt"
        with open(f"{os.getcwd()}/output/{filename}", "a") as f:
            f.write(
                f"{image_file_name} : Found {len(paths)} paths.\n"
            )

        return paths

    def optimize_paths(self, paths):
        """
        Replaced the nearest neighbor pathing algorithm.
        CCA already guarantees the order from Biggest to Smallest.
        """
        if not paths:
            return[]

        print("Using size-ordered paths from Connected Components (Biggest to Smallest)...")
        
        # Pre-convert all paths to float arrays to avoid type errors during streaming
        self.ordered_paths = [p.astype(float) for p in paths]

        return self.ordered_paths

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
    def __init__(self, ip_address, speed=100, acceleration=2000, z_draw=0, z_travel=20, grip_width=270):
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
        self.arm.set_servo_angle(servo_id=5, angle=90,
                                 is_radian=False, wait=True)
        self.arm.set_gripper_position(self.grip_width+50, wait=True)
        print(">>> PLEASE INSERT PEN NOW. You have 10 seconds. <<<")
        time.sleep(10)
        self.arm.set_gripper_position(
            self.grip_width-30, wait=True)  # Grip tight
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
                    self.arm.set_position(
                        z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 0
                self.arm.set_position(
                    x=target_x, y=target_y, z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)

            elif cmd_type == 1:  # DRAW (Pen Down)
                if current_state == 0:
                    self.arm.set_position(
                        z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 1
                # wait=False allows continuous motion (smoother curves)
                self.arm.set_position(
                    x=target_x, y=target_y, z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=False)

        # Lift at end
        self.arm.set_position(
            z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
        self.go_home()
        self.arm.disconnect()
        print("Drawing complete.")


class RemoteController:
    """
    A wrapper class for the configurations to run the XArmArtist and the image processing
    """
    def __init__(self, image_file=None):
        self.ROBOT_IP = "192.168.1.227"

        self.ROBOT_ORIGIN_X = 250
        self.ROBOT_ORIGIN_Y = -100
        self.DRAW_WIDTH_MM = 400

        self.GRIPPER_DEPTH = 74.4
        self.PEN_LENGTH = 128.4
        self.PEN_DOWN_Z = self.PEN_LENGTH - self.GRIPPER_DEPTH
        self.PEN_UP_Z = self.PEN_DOWN_Z + 15

        self.IMAGE_FILE = None
        self.pathplanner = None
        self.stream = None

        # Process an initial image if provided
        if image_file:
            self.load_image(image_file)

    def load_image(self, image_file):
        """Processes a new image and updates the drawing paths."""
        self.IMAGE_FILE = image_file
        self.pathplanner = PathPlanner(self.IMAGE_FILE, canvas_width_mm=self.DRAW_WIDTH_MM)
        
        try:
            print(f"Processing image: {self.IMAGE_FILE}...")
            raw_paths = self.pathplanner.process_portrait()
            self.pathplanner.optimize_paths(raw_paths)
            
            # Note: converting the generator to a list so it can be reused
            # for both simulation and real robot drawing without needing to re-process!
            self.stream = list(self.pathplanner.stream_data())
            print("Image processing complete and paths saved.")
        except Exception as e:
            print(f"Error processing image: {e}")
            self.stream = None

    def change_robot_config(self, **kwargs):
        self.ROBOT_ORIGIN_X = kwargs.get("ROBOT_ORIGIN_X", self.ROBOT_ORIGIN_X)
        self.ROBOT_ORIGIN_Y = kwargs.get("ROBOT_ORIGIN_Y", self.ROBOT_ORIGIN_Y)
        self.DRAW_WIDTH_MM = kwargs.get("DRAW_WIDTH_MM", self.DRAW_WIDTH_MM)

        self.GRIPPER_DEPTH = kwargs.get("GRIPPER_DEPTH", self.GRIPPER_DEPTH)
        self.PEN_LENGTH = kwargs.get("PEN_LENGTH", self.PEN_LENGTH)
        self.PEN_DOWN_Z = self.PEN_LENGTH - self.GRIPPER_DEPTH
        self.PEN_UP_Z = self.PEN_DOWN_Z + 15

    def run_simulation(self):
        if not self.stream:
            print("No drawing data available. Please load a valid image first.")
            return

        print("Running Simulation...")
        ink_x, ink_y = [],[]
        travel_x, travel_y = [],[]
        last_x, last_y = 0, 0

        for cmd, x, y in self.stream:
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
        plt.title(f"Simulation: {self.IMAGE_FILE}")
        plt.plot(travel_x, travel_y, 'r:', linewidth=0.3, alpha=0.3)
        plt.plot(ink_x, ink_y, 'k-', linewidth=0.8)  # Black ink
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.show()

    def run_robot_drawing(self):
        if not self.stream:
            print("No drawing data available. Please load a valid image first.")
            return

        print("Running on Real xArm...")
        import time

        start = time.time()
        bot = XArmArtist(self.ROBOT_IP, speed=300, acceleration=1000,
                         z_draw=self.PEN_DOWN_Z, z_travel=self.PEN_UP_Z, grip_width=270)
        bot.connect()
        bot.put_pen_in()
        bot.draw(self.stream, self.ROBOT_ORIGIN_X, self.ROBOT_ORIGIN_Y)
        print("Time Taken:")
        print(time.time() - start)


if __name__ == "__main__":
    remote = RemoteController()

    while True:
        print("\n=== xArm Artist Control Menu ===")
        print("0. Select image via File Explorer (USB/Phone)")
        print("1. Load new image (Enter path manually)")
        print("2. Run simulation")
        print("3. Run robot drawing")
        print("4. Exit")
        
        choice = input("Select an option (0-4): ").strip()
        
        if choice == '0':
            received_path = select_file_via_explorer()
            if received_path:
                print(f"Selected file: {received_path}")
                remote.load_image(received_path)
            else:
                print("No file selected.")
            
        elif choice == '1':
            img_path = input("Enter the file path for the image: ").strip()
            img_path = img_path.strip('\'"')
            remote.load_image(img_path)
            
        elif choice == '2':
            remote.run_simulation()
            
        elif choice == '3':
            confirm = input("Are you sure you want to send this to the real robot? (y/n): ").lower()
            if confirm == 'y':
                remote.run_robot_drawing()
            else:
                print("Robot run cancelled.")
                
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 0, 1, 2, 3, or 4.")
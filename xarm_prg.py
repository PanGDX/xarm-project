import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from xarm.wrapper import XArmAPI
from datetime import datetime
from rembg import remove
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

def select_file_via_explorer():
    """Opens a standard Windows file explorer dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("Opening file explorer...")
    file_path = filedialog.askopenfilename(
        title="Select an Image from USB/Phone",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path

class PathPlanner:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img_w = 0
        self.img_h = 0
        self.ordered_paths = []

    def display_step(self, img, title="Debug Step", save=False):
        """Helper to visualize image processing steps using Matplotlib and optionally save them."""
        if save:
            os.makedirs("output", exist_ok=True)
            safe_title = "".join([c for c in title if c.isalnum() or c == ' ']).rstrip()
            image_file_name = Path(self.image_path).name
            now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            filename = f"output/{image_file_name} {safe_title.replace(' ', '_').lower() + ' ' + now}.png"

            cv2.imwrite(filename, img)
            print(f"[*] Saved debug image: {filename}")

        plt.figure(figsize=(10, 8))
        plt.title(title)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()  

    def process_portrait(self):
        """Optimized for Faces and Fine Details with Visual Debugging."""
        img_color = cv2.imread(self.image_path)
        if img_color is None:
            raise ValueError("Image not found")

        img_nobg = remove(img_color)
        img = cv2.cvtColor(img_nobg, cv2.COLOR_BGRA2GRAY)

        # Resize maintaining aspect ratio
        h, w = img.shape
        new_w = 800
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)
        img = cv2.resize(img, (new_w, new_h))
        
        # Save exact pixel dimensions for the Best-Fit algorithm
        self.img_w = new_w
        self.img_h = new_h

        self.display_step(img, "1. Resized Image, removed background", save=True)

        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)
        self.display_step(img, "3. Bilateral Filter (Smoothed Skin)")

        edges = cv2.Canny(img, 20, 60)
        self.display_step(edges, "4a. Raw Canny Edges (1px wide)", save=True)

        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  
        binary_output = cv2.ximgproc.thinning(dilated)
        self.display_step(binary_output, "4b. Dilated (Thickened Lines)", save=True)

        # Connected Component Analysis + DFS
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_output, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        
        label_area_pairs = [(label, areas[label]) for label in range(1, num_labels)]
        label_area_pairs.sort(key=lambda x: x[1], reverse=True)
        
        y_coords, x_coords = np.nonzero(labels)
        label_vals = labels[y_coords, x_coords]
        
        component_pixels_dict = defaultdict(set)
        for y, x, lbl in zip(y_coords, x_coords, label_vals):
            component_pixels_dict[lbl].add((y, x))

        min_path_length = 15 
        paths = []
        debug_contours_canvas = np.zeros_like(img)

        print(f"Found {num_labels - 1} connected components. Running DFS to generate paths...")

        for label, area in label_area_pairs:
            if area < min_path_length:
                continue
                
            component_pixels = component_pixels_dict[label]
            
            start_node = None
            for node in component_pixels:
                r, c = node
                neighbors = [(r-1, c-1), (r-1, c), (r-1, c+1), 
                             (r, c-1),             (r, c+1), 
                             (r+1, c-1), (r+1, c), (r+1, c+1)]
                count = sum(1 for n in neighbors if n in component_pixels)
                if count == 1:
                    start_node = node
                    break
            
            if start_node is None:
                start_node = next(iter(component_pixels))
                
            stack = [start_node]
            visited = {start_node}
            path = []
            
            while stack:
                curr = stack[-1]
                if not path or path[-1] != curr:
                    path.append(curr)
                    
                r, c = curr
                neighbors = [(r-1, c-1), (r-1, c), (r-1, c+1), 
                             (r, c-1),             (r, c+1), 
                             (r+1, c-1), (r+1, c), (r+1, c+1)]
                             
                unvisited_neighbors = [n for n in neighbors if n in component_pixels and n not in visited]
                
                if unvisited_neighbors:
                    next_node = unvisited_neighbors[0]
                    visited.add(next_node)
                    stack.append(next_node)
                else:
                    stack.pop() 
                    if stack:
                        path.append(stack[-1]) 
                        
            path_xy = np.array([(c, r) for r, c in path], dtype=np.float32)
            epsilon = 1.0
            approx = cv2.approxPolyDP(path_xy, epsilon, closed=False)
            
            if len(approx) > 1:
                final_path = approx.reshape(-1, 2)
                paths.append(final_path)
                cv2.polylines(debug_contours_canvas, [np.int32(final_path)], isClosed=False, color=255, thickness=1)

        self.display_step(debug_contours_canvas, "5. Final DFS Paths", save=True)
        print(f"Details extracted. Found {len(paths)} paths.")

        return paths

    def optimize_paths(self, paths):
        if not paths:
            return []
        print("Using size-ordered paths from Connected Components (Biggest to Smallest)...")
        self.ordered_paths = [p.astype(float) for p in paths]
        return self.ordered_paths


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
        self.arm.set_servo_angle(servo_id=5, angle=90, is_radian=False, wait=True)
        self.arm.set_gripper_position(self.grip_width+50, wait=True)
        print(">>> PLEASE INSERT PEN NOW. You have 10 seconds. <<<")
        time.sleep(10)
        self.arm.set_gripper_position(self.grip_width-30, wait=True)
        time.sleep(1)

    def draw(self, data_stream):
        self.go_home()
        print("Starting drawing...")

        current_state = 0 
        for cmd_type, target_x, target_y in data_stream:
            # target_x and target_y are already mapped to real physical absolute coordinates
            if cmd_type == 0:  
                if current_state == 1:
                    self.arm.set_position(z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 0
                self.arm.set_position(x=target_x, y=target_y, z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)

            elif cmd_type == 1: 
                if current_state == 0:
                    self.arm.set_position(z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=True)
                    current_state = 1
                self.arm.set_position(x=target_x, y=target_y, z=self.z_draw, speed=self.speed, mvacc=self.acc, wait=False)

        self.arm.set_position(z=self.z_travel, speed=self.speed, mvacc=self.acc, wait=True)
        self.go_home()
        self.arm.disconnect()
        print("Drawing complete.")


class RemoteController:
    def __init__(self, image_file=None):
        self.ROBOT_IP = "192.168.1.227"

        # Hard bounds of the physical paper
        self.PAPER_MIN_X = 185
        self.PAPER_MAX_X = 400
        self.PAPER_MIN_Y = -100
        self.PAPER_MAX_Y = 75

        self.GRIPPER_DEPTH = 74.4
        self.PEN_LENGTH = 128.4
        self.PEN_DOWN_Z = self.PEN_LENGTH - self.GRIPPER_DEPTH
        self.PEN_UP_Z = self.PEN_DOWN_Z + 15

        self.IMAGE_FILE = None
        self.pathplanner = None
        self.stream = []

        if image_file:
            self.load_image(image_file)

    def load_image(self, image_file):
        """Processes image, tests orientations, and builds the safest/largest stream of robot absolute paths."""
        self.IMAGE_FILE = image_file
        self.pathplanner = PathPlanner(self.IMAGE_FILE)
        
        try:
            print(f"Processing image: {self.IMAGE_FILE}...")
            raw_paths = self.pathplanner.process_portrait()
            ordered_paths = self.pathplanner.optimize_paths(raw_paths)
            
            img_w = self.pathplanner.img_w
            img_h = self.pathplanner.img_h
            
            cx_pixel = img_w / 2.0
            cy_pixel = img_h / 2.0
            
            # --- CALCULATE BEST FIT (LANDSCAPE VS PORTRAIT) ---
            margin = 10  # mm buffer so we never draw on the exact edge of the paper
            min_x = self.PAPER_MIN_X + margin
            max_x = self.PAPER_MAX_X - margin
            min_y = self.PAPER_MIN_Y + margin
            max_y = self.PAPER_MAX_Y - margin
            
            avail_x = max_x - min_x  # Robot Depth
            avail_y = max_y - min_y  # Robot Lateral
            
            paper_cx = (max_x + min_x) / 2.0
            paper_cy = (max_y + min_y) / 2.0
            
            # Test 1: Upright mapping
            scale1_x = avail_x / img_h
            scale1_y = avail_y / img_w
            scale1 = min(scale1_x, scale1_y)
            
            # Test 2: Rotated 90 degrees mapping
            scale2_x = avail_x / img_w
            scale2_y = avail_y / img_h
            scale2 = min(scale2_x, scale2_y)
            
            if scale1 >= scale2:
                chosen_scale = scale1
                is_rotated = False
                print(f"[*] Orientation: Upright Portrait (Scale Factor: {chosen_scale:.4f})")
            else:
                chosen_scale = scale2
                is_rotated = True
                print(f"[*] Orientation: Rotated 90 Deg. for Best Fit (Scale Factor: {chosen_scale:.4f})")
                
            # Build physical stream
            self.stream = []
            for path in ordered_paths:
                if len(path) == 0: continue
                
                # Travel (Pen Up)
                pt_start = self._transform_point(path[0][0], path[0][1], cx_pixel, cy_pixel, paper_cx, paper_cy, chosen_scale, is_rotated)
                self.stream.append((0, pt_start[0], pt_start[1]))
                
                # Draw (Pen Down)
                for point in path:
                    pt = self._transform_point(point[0], point[1], cx_pixel, cy_pixel, paper_cx, paper_cy, chosen_scale, is_rotated)
                    self.stream.append((1, pt[0], pt[1]))
                    
            print("Image coordinates scaled and translated to Robot boundaries safely.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing image: {e}")
            self.stream = []

    def _transform_point(self, px, py, cx_pixel, cy_pixel, paper_cx, paper_cy, scale, is_rotated):
        """Translates pixel data into properly scaled physical robot data taking orientation into account."""
        px_centered = px - cx_pixel
        py_centered = py - cy_pixel
        
        if not is_rotated:
            # Upright: Image Y->Robot X (Depth), Image X->Robot Y (Lateral)
            robot_x = paper_cx - (py_centered * scale)
            robot_y = paper_cy - (px_centered * scale)
        else:
            # Rotated: Image X->Robot X (Depth), Image Y->Robot Y (Lateral)
            robot_x = paper_cx + (px_centered * scale)
            robot_y = paper_cy - (py_centered * scale)
            
        # Hard cap to bounds (Safety Fallback)
        robot_x = max(self.PAPER_MIN_X, min(self.PAPER_MAX_X, robot_x))
        robot_y = max(self.PAPER_MIN_Y, min(self.PAPER_MAX_Y, robot_y))
        
        return (robot_x, robot_y)

    def run_simulation(self):
        if not self.stream:
            print("No drawing data available. Please load a valid image first.")
            return

        print("Running Simulation...")
        ink_x, ink_y = [], []
        travel_x, travel_y = [], []
        last_vis_x, last_vis_y = None, None

        for cmd, target_x, target_y in self.stream:
            # Matplotlib X mapped to Robot Y (Left/Right)
            # Matplotlib Y mapped to Robot X (Depth)
            vis_x = target_y  
            vis_y = target_x  

            if last_vis_x is None:
                last_vis_x, last_vis_y = vis_x, vis_y

            if cmd == 0:
                travel_x.extend([last_vis_x, vis_x, np.nan])
                travel_y.extend([last_vis_y, vis_y, np.nan])
                ink_x.extend([np.nan])
                ink_y.extend([np.nan])
            elif cmd == 1:
                ink_x.extend([vis_x])
                ink_y.extend([vis_y])
            last_vis_x, last_vis_y = vis_x, vis_y

        plt.figure(figsize=(7, 9))
        plt.title(f"Desk View Perspective: {self.IMAGE_FILE}")
        
        # Paper Bounding Box mapped to visual view
        paper_vis_x = [self.PAPER_MIN_Y, self.PAPER_MAX_Y, self.PAPER_MAX_Y, self.PAPER_MIN_Y, self.PAPER_MIN_Y]
        paper_vis_y = [self.PAPER_MIN_X, self.PAPER_MIN_X, self.PAPER_MAX_X, self.PAPER_MAX_X, self.PAPER_MIN_X]
        plt.plot(paper_vis_x, paper_vis_y, 'g-', linewidth=2, label="Paper Edge Limit")

        plt.plot(travel_x, travel_y, 'r:', linewidth=0.3, alpha=0.3, label="Robot Travel")
        plt.plot(ink_x, ink_y, 'k-', linewidth=0.8, label="Pen Ink")
        
        plt.axis('equal')
        
        # Invert Matplotlib's X-Axis because robot's positive Y goes left
        # (This correctly fakes standing behind the robot arm looking at the desk)
        plt.gca().invert_xaxis() 
        
        plt.xlabel("Robot Base Y (mm) [Negative is Right, Positive is Left]")
        plt.ylabel("Robot Base X (mm) [Closer to Base <---> Farther Away]")
        plt.legend(loc="upper right")
        plt.show()

    def run_robot_drawing(self):
        if not self.stream:
            print("No drawing data available. Please load a valid image first.")
            return

        print("Running on Real xArm...")
        start = time.time()
        bot = XArmArtist(self.ROBOT_IP, speed=300, acceleration=1000,
                         z_draw=self.PEN_DOWN_Z, z_travel=self.PEN_UP_Z, grip_width=270)
        bot.connect()
        bot.put_pen_in()
        bot.draw(self.stream) # No offsets needed, they are already pre-calculated!
        print(f"Time Taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":  
    remote = RemoteController()

    while True:
        print("\n=== xArm Artist Control Menu ===")
        print("0. Select image via File Explorer (USB/Phone)")
        print("1. Load new image (Enter path manually)")
        print("2. Run simulation (Accurate Desk Perspective!)")
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

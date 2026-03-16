import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

# --- HELPER FUNCTION FOR PLOTTING ---
def show_step(image, title, is_vector=False, vector_data=None):
    plt.figure(figsize=(10, 8))
    
    if is_vector and vector_data is not None:
        # Plotting vector lines
        # Cycle colors to distinguish different strokes
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, path in enumerate(vector_data):
            c = colors[i % len(colors)]
            plt.plot(path[:, 0], path[:, 1], color=c, linewidth=1.5)
            # Plot start point
            plt.plot(path[0, 0], path[0, 1], 'o', color=c, markersize=3)
        
        plt.gca().invert_yaxis()
        plt.title(f"[FINAL] Greedy Erasure Paths\n{title}")
    else:
        # Plotting Pixel Data
        plt.imshow(image, cmap='gray')
        plt.title(title)

    plt.grid(which='major', color='#333333', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")
    plt.tight_layout()
    print(f"Displaying: {title} (Close window to continue...)")
    plt.show()

class HybridPathPlanner:
    def __init__(self, image_path):
        self.image_path = image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File {image_path} not found.")

    def count_pixel_neighbors(self, grid, x, y, h, w):
        """Counts 8-neighbors to determine if a pixel is an edge/endpoint."""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx] > 0:
                        count += 1
        return count

    def get_greedy_move(self, grid, curr_x, curr_y, dir_vec, h, w):
        """Finds the best next point based on angle (Dot Product)."""
        # The 8 offsets at distance 2
        print("Getting gritty")
        print(curr_x, curr_y)

        offsets = [
            (2, -2), (2, 0), (2, 2),
            (0, -2),         (0, 2),
            (-2, -2),(-2, 0),(-2, 2)
        ]

        candidates = []
        for dx, dy in offsets:
            nx, ny = curr_x + dx, curr_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if grid[ny, nx] > 0:
                    candidates.append((nx, ny))

        if not candidates:
            return None

        # If no previous direction, pick the first valid neighbor
        if dir_vec is None:
            return candidates[0]

        best_candidate = None
        max_cos_theta = -2.0 
        
        dir_norm = math.sqrt(dir_vec[0]**2 + dir_vec[1]**2)
        if dir_norm == 0: return candidates[0]

        for cx, cy in candidates:
            move_vec = (cx - curr_x, cy - curr_y)
            move_norm = math.sqrt(move_vec[0]**2 + move_vec[1]**2)
            
            # Dot Product
            dot = (dir_vec[0] * move_vec[0]) + (dir_vec[1] * move_vec[1])
            cos_theta = dot / (dir_norm * move_norm)
            
            if cos_theta > max_cos_theta:
                max_cos_theta = cos_theta
                best_candidate = (cx, cy)

        return best_candidate

    def run_greedy_erasure(self, binary_img):
        """
        The core algorithm:
        1. Find pixel with fewest neighbors (Endpoint).
        2. Trace line using dot product directionality.
        3. Erase path + neighbors.
        4. Repeat.
        """
        grid = binary_img.copy()
        h, w = grid.shape
        paths = []
        
        print("\n--- Running Greedy Erasure Algorithm ---")
        print("Scanning grid... (This may take some time)")

        while True:
            # --- 1. FIND START POINT (Fewest Neighbors) ---
            # Get all remaining white pixels
            points = np.argwhere(grid > 0)
            if len(points) == 0:
                break

            best_pt = None
            min_neighbors = 9 

            # Optimization: Check a subset if too large, or check all as requested
            # We check all points to strictly follow the "fewest neighbors" rule
            for r, c in points:
                n = self.count_pixel_neighbors(grid, c, r, h, w)
                if n < min_neighbors:
                    min_neighbors = n
                    best_pt = (c, r)
                    if min_neighbors <= 1: # Optimization: Can't get better than 1
                        break
            
            if best_pt is None: break

            # --- 2. TRACE PATH ---
            current_path = [best_pt]
            cx, cy = best_pt
            
            # Erase start (5x5 block)
            grid[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)] = 0
            
            current_dir = None

            while True:
                next_pt = self.get_greedy_move(grid, cx, cy, current_dir, h, w)
                
                if next_pt is None:
                    print("Lift")
                    break # Lift pen
                
                nx, ny = next_pt
                current_dir = (nx - cx, ny - cy)
                cx, cy = nx, ny
                
                current_path.append((cx, cy))
                
                # Erase path (5x5 block)
                grid[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)] = 0

            if len(current_path) > 1:
                paths.append(np.array(current_path))
                sys.stdout.write(f"\rPaths extracted: {len(paths)}")
                sys.stdout.flush()

        print("\nGreedy Erasure Complete.")
        return paths

    def process_pipeline(self):
        print(f"--- Processing {self.image_path} ---")

        # STEP 1: LOAD
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        show_step(img, "STEP 1: Original Image")

        # STEP 2: RESIZE
        # We use 300px width to keep the O(N^2) neighbor search strictly fast enough for a demo
        h, w = img.shape
        new_w = 300 
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)
        img_resized = cv2.resize(img, (new_w, new_h))
        show_step(img_resized, f"STEP 2: Resized ({new_w}x{new_h})")

        # STEP 3: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_resized)
        show_step(img_clahe, "STEP 3: CLAHE (Contrast)")

        # STEP 4: BLUR
        img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0) # Smaller blur for smaller image
        show_step(img_blur, "STEP 4: Gaussian Blur")

        # STEP 5: CANNY EDGE DETECTION
        # Canny works best for Greedy Erasure because it produces thin lines (1-2 px wide)
        v = np.median(img_blur)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(img_blur, lower, upper)
        
        # Dilate slightly to ensure connectivity for the algorithm
        kernel = np.ones((2,2), np.uint8)
        binary_output = cv2.dilate(edges, kernel, iterations=1)
        
        show_step(binary_output, "STEP 5: Canny Edges (Input for Erasure)")

        # STEP 6: GREEDY ERASURE (Replaces FindContours)
        vector_paths = self.run_greedy_erasure(binary_output)

        # STEP 7: FINAL RESULT
        show_step(None, f"Greedy Erasure Result: {len(vector_paths)} Strokes", 
                  is_vector=True, vector_data=vector_paths)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_FILE = "./test_img/cat.jpg" 
    
    # Generate dummy image if file missing
    if not os.path.exists(IMAGE_FILE):
        print("Image not found, creating test pattern...")
        os.makedirs("./test_img", exist_ok=True)
        dummy = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(dummy, (200, 200), 100, 255, -1)
        cv2.rectangle(dummy, (150, 150), (250, 250), 0, -1)
        cv2.imwrite(IMAGE_FILE, dummy)

    planner = HybridPathPlanner(IMAGE_FILE)
    planner.process_pipeline()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- HELPER FUNCTION FOR PLOTTING ---
def show_step(image, title, is_vector=False, vector_data=None):
    """
    Displays the current processing step with a grid.
    Blocks execution until the window is closed.
    """
    plt.figure(figsize=(10, 8))
    
    if is_vector and vector_data is not None:
        # Plotting vector lines (The robot path)
        for path in vector_data:
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1)
        
        # Invert Y to match image coordinate system (0,0 at top left)
        plt.gca().invert_yaxis()
        plt.title(f"[STEP 6] Final Vector Paths (Robot Motion)\n{title}")
    else:
        # Plotting Pixel Data (The computer vision)
        plt.imshow(image, cmap='gray')
        plt.title(title)

    # Add Grid for pixel coordinate reference
    plt.grid(which='major', color='#333333', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle=':', linewidth=0.2, alpha=0.5)
    
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")
    plt.tight_layout()
    
    print(f"Displaying: {title} (Close window to continue...)")
    plt.show()

class VisualPathPlanner:
    def __init__(self, image_path):
        self.image_path = image_path
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    def process_and_visualize(self, method="canny"):
        print(f"--- Starting Processing Pipeline [{method.upper()}] ---")

        # ---------------------------------------------------------
        # STEP 1: LOAD & GRAYSCALE
        # ---------------------------------------------------------
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        show_step(img, "STEP 1: Raw Loaded Image (Grayscale)")

        # ---------------------------------------------------------
        # STEP 2: RESIZE
        # ---------------------------------------------------------
        h, w = img.shape
        new_w = 800  # Target width
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        show_step(img_resized, f"STEP 2: Resized to {new_w}x{new_h} (Standardization)")

        # ---------------------------------------------------------
        # STEP 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # ---------------------------------------------------------
        # This equalizes light/dark areas so shadows become visible details
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_resized)
        
        show_step(img_clahe, "STEP 3: CLAHE Filter (Contrast Enhancement)")

        # ---------------------------------------------------------
        # STEP 4: GAUSSIAN BLUR
        # ---------------------------------------------------------
        # Removes high-frequency noise (skin texture, paper grain)
        img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        
        show_step(img_blur, "STEP 4: Gaussian Blur (Noise Reduction)")

        # ---------------------------------------------------------
        # STEP 5: EDGE DETECTION
        # ---------------------------------------------------------
        binary_output = None
        step_5_title = ""

        if method == "canny":
            # Dynamic thresholding based on median intensity
            v = np.median(img_blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            
            edges = cv2.Canny(img_blur, lower, upper)
            
            # Dilate to connect broken lines
            kernel = np.ones((2,2), np.uint8)
            binary_output = cv2.dilate(edges, kernel, iterations=1)
            step_5_title = f"STEP 5: Canny Edge Detection (Thresholds: {lower}-{upper}) + Dilation"
            
        elif method == "sketch":
            binary_output = cv2.adaptiveThreshold(
                img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 3
            )
            step_5_title = "STEP 5: Adaptive Thresholding (Sketch Style)"

        show_step(binary_output, step_5_title)

        # ---------------------------------------------------------
        # STEP 6: CONTOUR EXTRACTION (Vectorization)
        # ---------------------------------------------------------
        # Find contours returns a list of point arrays
        contours, _ = cv2.findContours(
            binary_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        filtered_paths = []
        min_path_length = 15 # Filter out small noise specs
        
        for cnt in contours:
            if cv2.arcLength(cnt, False) > min_path_length:
                # Simplify path (reduce point count for smoother robot motion)
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                filtered_paths.append(approx.reshape(-1, 2))

        # We visualize the vectors on a blank canvas to see what the robot actually "sees"
        show_step(None, f"Extracted {len(filtered_paths)} Vector Paths", is_vector=True, vector_data=filtered_paths)

        return filtered_paths

if __name__ == "__main__":
    # 1. Update this path to an image on your computer
    IMAGE_FILE = "./test_img/animegirl_bg_removed.jpg" 
    
    # 2. Choose Style: "canny" (clean lines) or "sketch" (messy shading)
    STYLE = "canny"

    # 3. Create defaults if file doesn't exist just to demonstrate
    if not os.path.exists(IMAGE_FILE):
        print(f"File {IMAGE_FILE} not found. Creating a dummy test image...")
        dummy = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(dummy, (250, 250), 100, 255, -1) # White circle
        cv2.rectangle(dummy, (100, 100), (400, 400), 255, 5) # White box
        os.makedirs("./test_img", exist_ok=True)
        cv2.imwrite(IMAGE_FILE, dummy)

    # 4. Run Process
    visualizer = VisualPathPlanner(IMAGE_FILE)
    visualizer.process_and_visualize(method=STYLE)
    
    print("--- Visualization Complete ---")
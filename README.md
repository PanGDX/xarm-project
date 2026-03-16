# Classes
PathPlanner takes in image. PathPlanner.process_portrait returns np.array of edges after processing.
and PathPlanner.optimize_paths uses a greedy path finder to make the pathing more optimized
then PathPlanner.stream_data() streams the information to XArm drawer


# Config Variables
    IMAGE_FILE = "./test_img/animegirl_bg_removed.jpg" 
    ROBOT_IP = "192.168.1.227"
    SIMULATION_MODE = True  # Set False to draw on real robot. Simulation shows matplotlib drawing. Approximately similar

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

    # Speed and acceleration of xarm can be modified in the XArm Class
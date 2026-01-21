#!/usr/bin/env python3
"""
Configuration file for CUDA-enabled AprilTag Detection
"""

# ============================================================================
# RTSP STREAM CONFIGURATION
# ============================================================================
RTSP_URL = "rtsps://10.45.41.253:7441/J4ZIcw3NhpITdOkM?enableSrtp"
RTSP_TRANSPORT = "tcp"  # tcp or udp
RTSP_RECONNECT_DELAY = 5  # seconds to wait before reconnecting on failure

# ============================================================================
# MJPEG STREAMER CONFIGURATION
# ============================================================================
# The detector fetches frames from the MJPEG streamer instead of opening its own RTSP connection
# This avoids CUDA memory corruption from multiple simultaneous RTSP connections
MJPEG_STREAMER_URL = "http://localhost:5002/snapshot"
MJPEG_STREAMER_TIMEOUT = 5  # seconds to wait for snapshot request

# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
USE_CUDA_PREPROCESSING = True  # Use GPU for image preprocessing

# ============================================================================
# APRILTAG DETECTION CONFIGURATION
# ============================================================================
# AprilTag family - using OpenCV's aruco module
# Available: DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11
APRILTAG_FAMILY = "DICT_APRILTAG_36h11"

# Detection parameters - aggressively tuned for maximum detection reliability
# These parameters are more permissive to catch tags that might be missed
DETECTOR_PARAMS = {
    # Adaptive thresholding - very wide range for varying tag sizes and lighting
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 103,  # Increased significantly for larger tags
    "adaptiveThreshWinSizeStep": 6,   # Larger steps to cover more ground
    "adaptiveThreshConstant": 5,      # Reduced from 7 - less aggressive threshold
    # Marker size constraints - very permissive
    "minMarkerPerimeterRate": 0.005,  # Very small tags allowed (was 0.01)
    "maxMarkerPerimeterRate": 8.0,    # Very large tags allowed (was 4.0)
    # Shape detection - more tolerant of imperfect shapes
    "polygonalApproxAccuracyRate": 0.08,  # Increased for more tolerance (was 0.05)
    "minCornerDistanceRate": 0.01,    # Reduced from 0.02
    "minDistanceToBorder": 0,         # Allow tags at edge of frame
    "minMarkerDistanceRate": 0.01,    # Reduced from 0.02
    # Corner refinement - use contour method for better results
    "cornerRefinementMethod": 2,      # CORNER_REFINE_CONTOUR
    "cornerRefinementWinSize": 7,     # Increased from 5 for larger refinement window
    "cornerRefinementMaxIterations": 100,  # Increased from 50
    "cornerRefinementMinAccuracy": 0.01,   # Reduced from 0.05 for better precision
    # Error correction - more permissive
    "errorCorrectionRate": 0.8,       # Increased from 0.6 - allow more bit errors
    # Perspective removal - helps with angled tags
    "perspectiveRemovePixelPerCell": 6,   # More samples per cell
    "perspectiveRemoveIgnoredMarginPerCell": 0.1,  # Smaller margin
    # Bit extraction
    "markerBorderBits": 1,
    # Detection mode
    "detectInvertedMarker": True,     # Also detect inverted (white on black) tags
}

# Enhanced detection - try multiple preprocessing methods
USE_CLAHE_ENHANCEMENT = True  # Contrast Limited Adaptive Histogram Equalization

# ============================================================================
# MQTT CONFIGURATION
# ============================================================================
MQTT_BROKER = "aadvncec0039959.ashleyfurniture.com"
MQTT_PORT = 1883
MQTT_TOPIC_DETECTIONS = "apriltag/teja/detections"
MQTT_TOPIC_STATS = "apriltag/teja/system_stats"
MQTT_QOS = 1
MQTT_CLIENT_ID = "apriltag_cuda_detector_teja"
MQTT_KEEPALIVE = 60

# ============================================================================
# FILE PATHS
# ============================================================================
# These paths are inside the container
IMAGE_SAVE_DIR = "/workspace/scripts/Apriltag_detect"
LATEST_FRAME_PATH = f"{IMAGE_SAVE_DIR}/latest_frame.jpg"
LATEST_DETECTED_PATH = f"{IMAGE_SAVE_DIR}/latest_detected.jpg"

# ============================================================================
# TIMING CONFIGURATION
# ============================================================================
DETECTION_INTERVAL = 5  # seconds between detections
STATS_PUBLISH_INTERVAL = 5  # seconds between system stats publishing

# ============================================================================
# CAMERA/SITE INFORMATION
# ============================================================================
CAMERA_LOCATION = "aadvca.0011.RouterWR255"
SITE_ID = "017"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# ANNOTATION CONFIGURATION
# ============================================================================
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 3
TEXT_COLOR = (0, 255, 0)  # Green
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black
TEXT_FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
TEXT_OUTLINE_THICKNESS = 4


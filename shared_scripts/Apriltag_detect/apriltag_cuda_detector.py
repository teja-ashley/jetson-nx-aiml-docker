#!/usr/bin/env python3
"""
CUDA-Enabled AprilTag Detector with MQTT Publishing
Uses OpenCV's built-in AprilTag detector with GPU acceleration
"""

import cv2
import numpy as np
import json
import time
import logging
import signal
import sys
import os
import requests
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import subprocess
import threading

# Import configuration
import config

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
running = True
mqtt_client = None
stats_lock = threading.Lock()
latest_stats = {}

# Cache CUDA device count to avoid repeated calls (prevents memory corruption)
CUDA_DEVICE_COUNT = None
CUDA_AVAILABLE = False

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================
def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global running
    sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
    logger.info(f"Shutdown signal received ({sig_name}). Cleaning up...")
    running = False

def sighup_handler(sig, frame):
    """Handle SIGHUP (SSH disconnection) - ignore it and keep running"""
    logger.info("SIGHUP received (SSH disconnection?). Ignoring and continuing...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, sighup_handler)  # Ignore SSH disconnection

# ============================================================================
# MQTT CALLBACKS
# ============================================================================
def on_connect(client, userdata, flags, rc, properties=None):
    """MQTT connection callback"""
    if rc == 0:
        logger.info(f"Connected to MQTT broker: {config.MQTT_BROKER}")
    else:
        logger.error(f"MQTT connection failed with code {rc}")

def on_disconnect(client, userdata, flags=None, rc=None, properties=None):
    """MQTT disconnection callback (compatible with paho-mqtt v2)"""
    # Handle both v1 (3 args) and v2 (5 args) signatures
    if rc is None and isinstance(flags, int):
        rc = flags  # v1 signature: flags is actually rc
    logger.warning(f"Disconnected from MQTT broker (code: {rc})")

def on_publish(client, userdata, mid, rc=None, properties=None):
    """MQTT publish callback (compatible with paho-mqtt v2)"""
    pass  # Silent success - avoid debug spam

# ============================================================================
# CUDA INITIALIZATION (Call once to avoid memory corruption)
# ============================================================================
def initialize_cuda():
    """Initialize CUDA and cache device count"""
    global CUDA_DEVICE_COUNT, CUDA_AVAILABLE

    try:
        CUDA_DEVICE_COUNT = cv2.cuda.getCudaEnabledDeviceCount()
        CUDA_AVAILABLE = CUDA_DEVICE_COUNT > 0
        logger.info(f"CUDA initialized: {CUDA_DEVICE_COUNT} device(s) available")
        return True
    except Exception as e:
        logger.warning(f"CUDA initialization failed: {e}")
        CUDA_DEVICE_COUNT = 0
        CUDA_AVAILABLE = False
        return False

# ============================================================================
# MQTT SETUP
# ============================================================================
def setup_mqtt():
    """Initialize MQTT client"""
    global mqtt_client

    try:
        mqtt_client = mqtt.Client(
            client_id=config.MQTT_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.on_publish = on_publish

        logger.info(f"Connecting to MQTT broker: {config.MQTT_BROKER}:{config.MQTT_PORT}")
        mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, config.MQTT_KEEPALIVE)
        mqtt_client.loop_start()
        time.sleep(2)  # Wait for connection

        return True
    except Exception as e:
        logger.error(f"MQTT setup failed: {e}")
        return False

# ============================================================================
# SYSTEM METRICS COLLECTION
# ============================================================================
def get_gpu_stats():
    """Get GPU statistics from Jetson thermal zones and /proc/meminfo"""
    gpu_util = 0
    gpu_temp = 0
    ram_used = 0
    ram_total = 0

    try:
        # Get GPU temperature from thermal zones
        thermal_zones = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/devices/virtual/thermal/thermal_zone1/temp',
            '/sys/devices/virtual/thermal/thermal_zone2/temp',
        ]

        for zone in thermal_zones:
            try:
                with open(zone, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0  # Convert millidegrees to degrees
                    if temp > gpu_temp:  # Use the highest temperature
                        gpu_temp = temp
            except:
                continue

        # Get RAM usage from /proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()

            # Parse MemTotal and MemAvailable
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    ram_total = int(line.split()[1]) // 1024  # Convert KB to MB
                elif 'MemAvailable:' in line:
                    ram_available = int(line.split()[1]) // 1024  # Convert KB to MB

            ram_used = ram_total - ram_available
        except Exception as e:
            logger.debug(f"Could not read /proc/meminfo: {e}")

        return {
            "gpu_utilization_percent": gpu_util,
            "gpu_temperature_c": round(gpu_temp, 1),
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total
        }
    except Exception as e:
        logger.debug(f"Could not get GPU stats: {e}")
        return {
            "gpu_utilization_percent": 0,
            "gpu_temperature_c": 0,
            "ram_used_mb": 0,
            "ram_total_mb": 0
        }

def collect_system_stats(processing_time_ms, detections_count):
    """Collect comprehensive system statistics"""
    global latest_stats, CUDA_AVAILABLE, CUDA_DEVICE_COUNT

    try:
        # Use cached CUDA values instead of calling getCudaEnabledDeviceCount()
        gpu_stats = get_gpu_stats()

        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "opencv_version": cv2.__version__,
            "cuda_available": CUDA_AVAILABLE,
            "cuda_devices": CUDA_DEVICE_COUNT,
            "processing_time_ms": round(processing_time_ms, 2),
            "fps": round(1000.0 / processing_time_ms, 2) if processing_time_ms > 0 else 0,
            "detections_count": detections_count,
            **gpu_stats
        }

        with stats_lock:
            latest_stats = stats

        return stats
    except Exception as e:
        logger.error(f"Error collecting system stats: {e}")
        return {}

# ============================================================================
# APRILTAG DETECTOR SETUP
# ============================================================================
def setup_apriltag_detector():
    """Initialize OpenCV AprilTag detector"""
    try:
        # Get AprilTag dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config.APRILTAG_FAMILY)
        )

        # Create detector parameters
        detector_params = cv2.aruco.DetectorParameters()

        # Set custom parameters from config
        for param_name, param_value in config.DETECTOR_PARAMS.items():
            if hasattr(detector_params, param_name):
                setattr(detector_params, param_name, param_value)

        # Create detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        logger.info(f"AprilTag detector initialized: {config.APRILTAG_FAMILY}")
        return detector
    except Exception as e:
        logger.error(f"Failed to initialize AprilTag detector: {e}")
        return None

# ============================================================================
# FRAME CAPTURE FROM MJPEG STREAMER
# ============================================================================
def fetch_frame_from_streamer():
    """Fetch a single frame from the MJPEG streamer's /snapshot endpoint.

    This approach uses the MJPEG streamer as the single source of RTSP frames,
    avoiding CUDA memory corruption from multiple simultaneous RTSP connections.

    Returns:
        tuple: (success: bool, frame: numpy.ndarray or None)
    """
    try:
        # Fetch JPEG from streamer
        response = requests.get(
            config.MJPEG_STREAMER_URL,
            timeout=config.MJPEG_STREAMER_TIMEOUT
        )

        if response.status_code != 200:
            logger.error(f"Streamer returned status {response.status_code}")
            return False, None

        # Decode JPEG to numpy array
        jpeg_bytes = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode JPEG from streamer")
            return False, None

        return True, frame

    except requests.exceptions.Timeout:
        logger.error("Timeout fetching frame from MJPEG streamer")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error - is MJPEG streamer running?")
        return False, None
    except Exception as e:
        logger.error(f"Error fetching frame from streamer: {e}")
        return False, None


def check_streamer_health():
    """Check if the MJPEG streamer is running and healthy."""
    try:
        health_url = config.MJPEG_STREAMER_URL.replace("/snapshot", "/health")
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            health = response.json()
            return health.get("has_frame", False)
        return False
    except Exception:
        return False

# ============================================================================
# CUDA IMAGE PROCESSING
# ============================================================================
def preprocess_frame_cuda(frame):
    """Preprocess frame using CUDA acceleration"""
    global CUDA_AVAILABLE

    # Use cached CUDA_AVAILABLE instead of checking device count
    if not CUDA_AVAILABLE:
        return preprocess_frame_cpu(frame)

    try:
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Resize on GPU if needed
        if frame.shape[1] != config.TARGET_WIDTH or frame.shape[0] != config.TARGET_HEIGHT:
            gpu_resized = cv2.cuda.resize(gpu_frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT))
        else:
            gpu_resized = gpu_frame

        # Convert to grayscale on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)

        # Download results
        gray = gpu_gray.download()
        resized = gpu_resized.download()

        return resized, gray

    except Exception as e:
        logger.error(f"CUDA preprocessing failed: {e}")
        # Fallback to CPU
        return preprocess_frame_cpu(frame)

def preprocess_frame_cpu(frame):
    """Preprocess frame using CPU (fallback)"""
    # Resize if needed
    if frame.shape[1] != config.TARGET_WIDTH or frame.shape[0] != config.TARGET_HEIGHT:
        resized = cv2.resize(frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT))
    else:
        resized = frame

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return resized, gray

# ============================================================================
# CUDA-ACCELERATED IMAGE PREPROCESSING
# ============================================================================

# Global CUDA objects (initialized once to avoid repeated allocation)
cuda_clahe = None
cuda_clahe_strong = None
cuda_gaussian_filter = None
cuda_default_stream = None

def initialize_cuda_filters():
    """Initialize CUDA filters once for reuse"""
    global cuda_clahe, cuda_clahe_strong, cuda_gaussian_filter, cuda_default_stream

    if not CUDA_AVAILABLE:
        return False

    try:
        # Create CLAHE on GPU
        cuda_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cuda_clahe_strong = cv2.cuda.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

        # Create Gaussian filter for unsharp masking (5x5 kernel)
        cuda_gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.0
        )

        # Use a regular CUDA stream instead of Stream_Null() to avoid memory corruption
        # Stream_Null() causes "double free or corruption" crashes during garbage collection
        cuda_default_stream = cv2.cuda_Stream()

        logger.info("CUDA filters initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CUDA filters: {e}")
        return False

def preprocess_cuda_parallel(gpu_gray):
    """
    Run multiple preprocessing methods on GPU.
    Returns list of (name, cpu_frame) tuples.
    API: clahe.apply(src, stream[, dst]) -> dst
    """
    global cuda_clahe, cuda_clahe_strong, cuda_gaussian_filter, cuda_default_stream

    results = []

    try:
        # CLAHE enhancement on GPU - correct API: apply(src, stream) -> dst
        gpu_clahe = cuda_clahe.apply(gpu_gray, cuda_default_stream)
        cuda_default_stream.waitForCompletion()  # Sync before download to avoid memory corruption
        results.append(("clahe", gpu_clahe.download()))

        # Strong CLAHE enhancement on GPU
        gpu_clahe_strong_result = cuda_clahe_strong.apply(gpu_gray, cuda_default_stream)
        cuda_default_stream.waitForCompletion()  # Sync before download
        results.append(("clahe_strong", gpu_clahe_strong_result.download()))

        # Unsharp masking (sharpening) on GPU
        # Get dimensions from input - size() returns (width, height)
        input_size = gpu_gray.size()
        input_type = gpu_gray.type()

        # Create output GpuMat with same size/type as input
        # Note: cv2.cuda_GpuMat constructor takes (rows, cols, type) or (size, type)
        # where size is (width, height)
        gpu_blurred = cv2.cuda_GpuMat(input_size, input_type)
        cuda_gaussian_filter.apply(gpu_gray, gpu_blurred)

        # Verify sizes match before addWeighted
        if gpu_gray.size() != gpu_blurred.size():
            logger.error(f"Size mismatch: gpu_gray={gpu_gray.size()}, gpu_blurred={gpu_blurred.size()}")
        else:
            # sharpened = original * 1.5 - blurred * 0.5
            gpu_sharpened = cv2.cuda.addWeighted(gpu_gray, 1.5, gpu_blurred, -0.5, 0)
            results.append(("sharpened", gpu_sharpened.download()))

    except Exception as e:
        logger.error(f"CUDA preprocessing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return results

def preprocess_cuda_advanced(gpu_gray):
    """
    Advanced GPU preprocessing with industrial-grade filters.
    Uses denoising and edge enhancement.
    """
    global cuda_clahe, cuda_default_stream
    results = []

    try:
        # Fast Non-Local Means Denoising on GPU (industrial-grade denoising)
        gpu_denoised = cv2.cuda.fastNlMeansDenoising(gpu_gray, h=10)

        # Apply CLAHE to denoised image - correct API: apply(src, stream) -> dst
        gpu_denoised_clahe = cuda_clahe.apply(gpu_denoised, cuda_default_stream)
        cuda_default_stream.waitForCompletion()  # Sync before download
        results.append(("denoised_clahe", gpu_denoised_clahe.download()))

        # Bilateral filter - edge-preserving smoothing (good for noisy images)
        gpu_bilateral = cv2.cuda.bilateralFilter(gpu_gray, 9, 75, 75)
        gpu_bilateral_clahe = cuda_clahe.apply(gpu_bilateral, cuda_default_stream)
        cuda_default_stream.waitForCompletion()  # Sync before download
        results.append(("bilateral_clahe", gpu_bilateral_clahe.download()))

    except Exception as e:
        logger.debug(f"Advanced CUDA preprocessing failed: {e}")

    return results

# ============================================================================
# APRILTAG DETECTION WITH CUDA PREPROCESSING
# ============================================================================
def detect_apriltags(detector, gray_frame, use_clahe=None):
    """
    Detect AprilTags using CUDA-accelerated parallel preprocessing.
    All image enhancement runs on GPU simultaneously.
    """
    global CUDA_AVAILABLE

    try:
        all_detections = {}

        if CUDA_AVAILABLE and cuda_clahe is not None:
            # Upload frame to GPU once
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray_frame)

            # Run parallel preprocessing on GPU
            preprocessed_frames = preprocess_cuda_parallel(gpu_gray)

            # Add original frame
            preprocessed_frames.insert(0, ("original", gray_frame))

            # Try detection on each preprocessed frame
            for method_name, frame in preprocessed_frames:
                corners, ids, rejected = detector.detectMarkers(frame)

                if ids is not None and len(ids) > 0:
                    _add_detections(all_detections, corners, ids, method_name)
                    # Early exit if we found tags with first few methods
                    if len(all_detections) > 0 and method_name in ["clahe", "original"]:
                        break

            # If no detection yet, try advanced preprocessing
            if len(all_detections) == 0:
                advanced_frames = preprocess_cuda_advanced(gpu_gray)
                for method_name, frame in advanced_frames:
                    corners, ids, rejected = detector.detectMarkers(frame)
                    if ids is not None and len(ids) > 0:
                        _add_detections(all_detections, corners, ids, method_name)
                        break
        else:
            # CPU fallback
            corners, ids, rejected = detector.detectMarkers(gray_frame)
            if ids is not None and len(ids) > 0:
                _add_detections(all_detections, corners, ids, "cpu_original")

        if all_detections:
            methods = set(d["detection_method"] for d in all_detections.values())
            logger.info(f"Detected {len(all_detections)} tags using: {methods}")

        return list(all_detections.values())

    except Exception as e:
        logger.error(f"AprilTag detection failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []

def _add_detections(all_detections, corners, ids, method_name):
    """Helper to add detections to results dict (deduplicates by tag_id)"""
    for i, tag_id in enumerate(ids):
        tid = int(tag_id[0])
        if tid not in all_detections:
            tag_corners = corners[i][0]
            center_x = int(np.mean(tag_corners[:, 0]))
            center_y = int(np.mean(tag_corners[:, 1]))

            all_detections[tid] = {
                "tag_id": tid,
                "center": [center_x, center_y],
                "corners": tag_corners.tolist(),
                "camera_location": config.CAMERA_LOCATION,
                "site_id": config.SITE_ID,
                "detection_method": method_name
            }

# ============================================================================
# IMAGE ANNOTATION
# ============================================================================
def annotate_frame(frame, detections):
    """Draw bounding boxes, outer highlight, IDs, and timestamp on frame"""
    annotated = frame.copy()

    # Add timestamp at top-left corner if there are detections
    if len(detections) > 0:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_text = f"Detected: {timestamp_str}"

        # Background rectangle for better visibility
        text_size = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)

        # Timestamp text
        cv2.putText(
            annotated, timestamp_text, (10, text_size[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2  # Cyan color for timestamp
        )

    for detection in detections:
        tag_id = detection["tag_id"]
        corners = np.array(detection["corners"], dtype=np.int32)
        center = detection["center"]

        # Calculate bounding rectangle for outer highlight
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))

        # Outer highlight box - larger rectangle with padding
        padding = 20
        outer_top_left = (max(0, min_x - padding), max(0, min_y - padding))
        outer_bottom_right = (min(frame.shape[1], max_x + padding), min(frame.shape[0], max_y + padding))

        # Draw outer highlight box (thicker, different color - cyan)
        cv2.rectangle(annotated, outer_top_left, outer_bottom_right, (255, 255, 0), 4)

        # Draw corner arrows/markers pointing to the tag
        arrow_length = 15
        arrow_color = (0, 255, 255)  # Yellow

        # Top-left corner arrow
        cv2.line(annotated, outer_top_left, (outer_top_left[0] + arrow_length, outer_top_left[1]), arrow_color, 3)
        cv2.line(annotated, outer_top_left, (outer_top_left[0], outer_top_left[1] + arrow_length), arrow_color, 3)

        # Top-right corner arrow
        cv2.line(annotated, (outer_bottom_right[0], outer_top_left[1]),
                 (outer_bottom_right[0] - arrow_length, outer_top_left[1]), arrow_color, 3)
        cv2.line(annotated, (outer_bottom_right[0], outer_top_left[1]),
                 (outer_bottom_right[0], outer_top_left[1] + arrow_length), arrow_color, 3)

        # Bottom-left corner arrow
        cv2.line(annotated, (outer_top_left[0], outer_bottom_right[1]),
                 (outer_top_left[0] + arrow_length, outer_bottom_right[1]), arrow_color, 3)
        cv2.line(annotated, (outer_top_left[0], outer_bottom_right[1]),
                 (outer_top_left[0], outer_bottom_right[1] - arrow_length), arrow_color, 3)

        # Bottom-right corner arrow
        cv2.line(annotated, outer_bottom_right,
                 (outer_bottom_right[0] - arrow_length, outer_bottom_right[1]), arrow_color, 3)
        cv2.line(annotated, outer_bottom_right,
                 (outer_bottom_right[0], outer_bottom_right[1] - arrow_length), arrow_color, 3)

        # Draw inner bounding box (original polygon)
        cv2.polylines(annotated, [corners], True, config.BBOX_COLOR, config.BBOX_THICKNESS)

        # Draw center point
        cv2.circle(annotated, tuple(center), 5, (0, 0, 255), -1)

        # Draw tag ID with outline - positioned above the outer box
        text = f"ID:{tag_id}"
        text_pos = (outer_top_left[0], outer_top_left[1] - 10)

        # Ensure text position is within frame
        if text_pos[1] < 20:
            text_pos = (outer_top_left[0], outer_bottom_right[1] + 25)

        # Black outline
        cv2.putText(
            annotated, text, text_pos,
            config.TEXT_FONT, config.TEXT_SCALE + 0.3,  # Slightly larger text
            config.TEXT_OUTLINE_COLOR, config.TEXT_OUTLINE_THICKNESS + 1
        )

        # Green text
        cv2.putText(
            annotated, text, text_pos,
            config.TEXT_FONT, config.TEXT_SCALE + 0.3,
            config.TEXT_COLOR, config.TEXT_THICKNESS
        )

    return annotated

# ============================================================================
# MQTT PUBLISHING
# ============================================================================
def publish_detections(detections, image_path):
    """Publish detection data to MQTT"""
    global mqtt_client

    if mqtt_client is None or not mqtt_client.is_connected():
        logger.warning("MQTT client not connected. Skipping publish.")
        return

    try:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "image_path": image_path,
            "detections": detections,
            "count": len(detections)
        }

        mqtt_client.publish(
            config.MQTT_TOPIC_DETECTIONS,
            json.dumps(payload),
            qos=config.MQTT_QOS
        )

        logger.info(f"Published {len(detections)} detections to MQTT")

    except Exception as e:
        logger.error(f"Failed to publish detections: {e}")

def publish_system_stats(stats):
    """Publish system statistics to MQTT"""
    global mqtt_client

    if mqtt_client is None or not mqtt_client.is_connected():
        logger.warning("MQTT client not connected. Skipping stats publish.")
        return

    try:
        mqtt_client.publish(
            config.MQTT_TOPIC_STATS,
            json.dumps(stats),
            qos=config.MQTT_QOS
        )

        logger.debug(f"Published system stats to MQTT")

    except Exception as e:
        logger.error(f"Failed to publish system stats: {e}")

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def process_frame(detector):
    """Fetch frame from MJPEG streamer and process for AprilTag detection."""
    global CUDA_AVAILABLE

    start_time = time.time()

    try:
        # Fetch frame from MJPEG streamer (no direct RTSP connection)
        ret, frame = fetch_frame_from_streamer()

        if not ret or frame is None:
            logger.error("Failed to fetch frame from streamer")
            return False

        fetch_time_ms = (time.time() - start_time) * 1000

        # Frame is already resized by streamer, but convert to grayscale for detection
        if config.USE_CUDA_PREPROCESSING and CUDA_AVAILABLE:
            resized_frame, gray_frame = preprocess_frame_cuda(frame)
        else:
            resized_frame, gray_frame = preprocess_frame_cpu(frame)

        # Detect AprilTags
        detections = detect_apriltags(detector, gray_frame)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Collect system stats
        stats = collect_system_stats(processing_time_ms, len(detections))

        # Save raw frame
        cv2.imwrite(config.LATEST_FRAME_PATH, resized_frame)

        # Annotate and save detected frame
        if len(detections) > 0:
            annotated_frame = annotate_frame(resized_frame, detections)
            cv2.imwrite(config.LATEST_DETECTED_PATH, annotated_frame)
            logger.info(f"Detected {len(detections)} AprilTags (fetch: {fetch_time_ms:.0f}ms, total: {processing_time_ms:.0f}ms)")

            # Publish detections
            publish_detections(detections, config.LATEST_DETECTED_PATH)
        else:
            # No detections - save original frame
            cv2.imwrite(config.LATEST_DETECTED_PATH, resized_frame)
            logger.info(f"No AprilTags detected (fetch: {fetch_time_ms:.0f}ms, total: {processing_time_ms:.0f}ms)")

            # Publish empty detections
            publish_detections([], config.LATEST_FRAME_PATH)

        # Publish system stats
        publish_system_stats(stats)

        return True

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main execution function - Fetches frames from MJPEG streamer."""
    global running, mqtt_client, CUDA_AVAILABLE, CUDA_DEVICE_COUNT

    logger.info("=" * 80)
    logger.info("AprilTag Detector Starting (MJPEG Streamer Mode)...")
    logger.info("=" * 80)
    logger.info(f"OpenCV Version: {cv2.__version__}")

    # Initialize CUDA once at startup
    initialize_cuda()
    logger.info(f"CUDA Devices: {CUDA_DEVICE_COUNT} (Available: {CUDA_AVAILABLE})")

    # Initialize CUDA filters for parallel preprocessing
    if CUDA_AVAILABLE:
        if initialize_cuda_filters():
            logger.info("CUDA parallel preprocessing enabled")
        else:
            logger.warning("CUDA filters failed to initialize, using CPU fallback")

    logger.info(f"MJPEG Streamer URL: {config.MJPEG_STREAMER_URL}")
    logger.info(f"Target Resolution: {config.TARGET_WIDTH}x{config.TARGET_HEIGHT}")
    logger.info(f"Detection Interval: {config.DETECTION_INTERVAL}s")
    logger.info(f"MQTT Broker: {config.MQTT_BROKER}:{config.MQTT_PORT}")
    logger.info(f"MQTT Topics: {config.MQTT_TOPIC_DETECTIONS}, {config.MQTT_TOPIC_STATS}")
    logger.info("=" * 80)

    # Setup MQTT
    if not setup_mqtt():
        logger.error("Failed to setup MQTT. Exiting.")
        return 1

    # Setup AprilTag detector
    detector = setup_apriltag_detector()
    if detector is None:
        logger.error("Failed to setup AprilTag detector. Exiting.")
        return 1

    # Wait for MJPEG streamer to be available
    logger.info("Waiting for MJPEG streamer...")
    streamer_ready = False
    for attempt in range(30):  # Wait up to 30 seconds
        if check_streamer_health():
            streamer_ready = True
            logger.info("MJPEG streamer is ready")
            break
        logger.info(f"Waiting for streamer... ({attempt + 1}/30)")
        time.sleep(1)

    if not streamer_ready:
        logger.error("MJPEG streamer not available. Make sure mjpeg_streamer.py is running.")
        return 1

    # Test frame fetch
    logger.info("Testing frame fetch from streamer...")
    ret, test_frame = fetch_frame_from_streamer()
    if ret and test_frame is not None:
        logger.info(f"Frame fetch OK. Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    else:
        logger.warning("Initial frame fetch failed. Will retry in main loop.")

    logger.info("Starting detection loop...")

    # Main loop
    frame_count = 0
    error_count = 0
    max_errors = 10
    exit_code = 0  # Default exit code

    try:
        while running:
            loop_start = time.time()

            # Process frame (fetched from MJPEG streamer)
            success = process_frame(detector)

            if success:
                frame_count += 1
                error_count = 0
            else:
                error_count += 1
                logger.warning(f"Frame processing failed ({error_count}/{max_errors})")

                if error_count >= max_errors:
                    logger.error("Too many consecutive errors. Check MJPEG streamer.")
                    error_count = 0
                    time.sleep(config.RTSP_RECONNECT_DELAY)

            # Wait for next interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, config.DETECTION_INTERVAL - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        exit_code = 130  # Standard exit code for SIGINT (128 + 2)

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit_code = 1  # Error exit code

    else:
        # Normal exit from loop (running set to False by signal)
        exit_code = 143  # Standard exit code for SIGTERM (128 + 15)

    finally:
        # Cleanup
        logger.info("Shutting down...")

        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.info("MQTT client disconnected")

        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Shutdown complete with exit code: {exit_code}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())


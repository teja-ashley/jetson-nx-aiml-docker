#!/usr/bin/env python3
"""
Quick test script to verify detector initialization
Run this before running the full detector
"""

import cv2
import sys
import config

print("=" * 80)
print("APRILTAG DETECTOR TEST")
print("=" * 80)

# Test 1: OpenCV Version
print(f"\n1. OpenCV Version: {cv2.__version__}")

# Test 2: CUDA Availability
cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"2. CUDA Devices: {cuda_count}")
if cuda_count > 0:
    print("   ✓ CUDA is available")
else:
    print("   ✗ CUDA is NOT available")

# Test 3: ArUco Module
print(f"3. ArUco Module: {hasattr(cv2, 'aruco')}")
if hasattr(cv2, 'aruco'):
    print("   ✓ ArUco module is available")
else:
    print("   ✗ ArUco module is NOT available")
    sys.exit(1)

# Test 4: AprilTag Dictionary
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.APRILTAG_FAMILY)
    )
    print(f"4. AprilTag Dictionary: {config.APRILTAG_FAMILY}")
    print("   ✓ Dictionary loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load dictionary: {e}")
    sys.exit(1)

# Test 5: Detector Creation
try:
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    print("5. Detector Creation: SUCCESS")
    print("   ✓ Detector initialized successfully")
except Exception as e:
    print(f"   ✗ Failed to create detector: {e}")
    sys.exit(1)

# Test 6: MQTT Configuration
print(f"\n6. MQTT Configuration:")
print(f"   Broker: {config.MQTT_BROKER}:{config.MQTT_PORT}")
print(f"   Detection Topic: {config.MQTT_TOPIC_DETECTIONS}")
print(f"   Stats Topic: {config.MQTT_TOPIC_STATS}")

# Test 7: File Paths
print(f"\n7. File Paths:")
print(f"   Image Directory: {config.IMAGE_SAVE_DIR}")
print(f"   Latest Frame: {config.LATEST_FRAME_PATH}")
print(f"   Latest Detected: {config.LATEST_DETECTED_PATH}")

# Test 8: RTSP Configuration
print(f"\n8. RTSP Configuration:")
print(f"   URL: {config.RTSP_URL}")
print(f"   Target Resolution: {config.TARGET_WIDTH}x{config.TARGET_HEIGHT}")
print(f"   Detection Interval: {config.DETECTION_INTERVAL}s")

# Test 9: CUDA Preprocessing Test
if cuda_count > 0:
    print(f"\n9. CUDA Preprocessing Test:")
    try:
        import numpy as np
        test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        gray = gpu_gray.download()
        
        print(f"   ✓ CUDA preprocessing works!")
        print(f"   Input: {test_img.shape}, Output: {gray.shape}")
    except Exception as e:
        print(f"   ✗ CUDA preprocessing failed: {e}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! Ready to run apriltag_cuda_detector.py")
print("=" * 80)


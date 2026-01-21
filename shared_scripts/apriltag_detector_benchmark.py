#!/usr/bin/env python3
"""
apriltag_detector_benchmark.py
Runs AprilTag detection on a pre-recorded video for benchmarking
"""

import time
import json
import cv2
import os
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from pupil_apriltags import Detector


## The video file has been downloaded from 01/08/2026 10:37:18 AM EST to 10:55:48 AM EST
# ============================================================================
# CONFIGURATION - BENCHMARK MODE
# ============================================================================

# Video file settings
VIDEO_PATH = "apriltag_detection1.mp4"
# VIDEO_PATH="/home/afi/Documents/test_folder/apriltag/videos/combined_video_files.mp4"
#VIDEO_PATH="/home/afi/Documents/test_folder/apriltag/videos/23_12_2025_one_cart_arrival.mp4"
BENCHMARK_MODE = True  # Set to False to use RTSP stream

# RTSP settings (used when BENCHMARK_MODE = False)
RTSP_URL = "rtsps://10.45.41.253:7441/J4ZIcw3NhpITdOkM?enableSrtp"

# Output paths
FRAME_PATH = "/workspace/scripts/liveframes/live_frame.jpg"
ANNOTATED_PATH = "/workspace/scripts/liveframes/live_frame_detected.jpg"
BENCHMARK_LOG = "/workspace/scripts/liveframes/benchmark_results.json"

# MQTT settings
MQTT_BROKER = "aadvncec0039959.ashleyfurniture.com"
MQTT_PORT = 1883
MQTT_TOPIC = "apriltag/benchmarkdocker"

# Processing settings
PROCESS_FPS = 1.0  # Process 1 frame per second from video
CAMERA_NAME = "aadvca.0011.RouterWR255"
SITE_ID = "017"

# ============================================================================
# GLOBALS
# ============================================================================

detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)
mqtt_client = None

# Benchmark statistics
benchmark_stats = {
    "video_path": VIDEO_PATH,
    "start_time": None,
    "end_time": None,
    "total_frames_processed": 0,
    "frames_with_detections": 0,
    "total_tags_detected": 0,
    "unique_tag_ids": set(),
    "avg_processing_time_ms": 0,
    "frame_processing_times": [],
    "detections_by_tag_id": {}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_utc_now():
    """Returns ISO 8601 UTC timestamp"""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def detect_tags(frame):
    """
    Run AprilTag detection on frame
    Returns: (list of detections, annotated frame)
    """
    if frame is None:
        return [], None
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Time the detection
    detect_start = time.time()
    detections = detector.detect(gray)
    detect_time_ms = (time.time() - detect_start) * 1000
    
    # Update benchmark stats
    benchmark_stats["frame_processing_times"].append(detect_time_ms)
    
    results = []
    annotated = frame.copy()
    
    for d in detections:
        tag_id = int(d.tag_id)
        
        results.append({
            "tag_id": tag_id,
            "confidence": float(d.decision_margin),
            "center": d.center.tolist(),
            "corners": d.corners.tolist(),
            "hamming": int(d.hamming)
        })
        
        # Update benchmark stats
        benchmark_stats["unique_tag_ids"].add(tag_id)
        benchmark_stats["detections_by_tag_id"][tag_id] = \
            benchmark_stats["detections_by_tag_id"].get(tag_id, 0) + 1
        
        # Draw bounding box
        corners = d.corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i+1) % 4])
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 3)
        
        # Draw tag ID with outline
        center = d.center.astype(int)
        text = f"ID:{tag_id}"
        
        # Black outline
        cv2.putText(annotated, text, (center[0] - 20, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        # Green text
        cv2.putText(annotated, text, (center[0] - 20, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Update stats
    if len(results) > 0:
        benchmark_stats["frames_with_detections"] += 1
        benchmark_stats["total_tags_detected"] += len(results)
    
    return results, annotated

def publish_to_mqtt(detections, frame_number):
    """
    Publish detection results to MQTT
    """
    payload = {
        "timestamp": get_utc_now(),
        "camera_name": CAMERA_NAME,
        "site_id": SITE_ID,
        "frame_number": frame_number,
        "benchmark_mode": True,
        "detections": detections
    }
    
    try:
        result = mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[{get_utc_now()}] ❌ MQTT publish failed (rc={result.rc})")
    
    except Exception as e:
        print(f"[{get_utc_now()}] ❌ MQTT error: {e}")

def save_benchmark_results():
    """
    Save benchmark statistics to JSON file
    """
    benchmark_stats["unique_tag_ids"] = list(benchmark_stats["unique_tag_ids"])
    
    # Calculate average processing time
    if benchmark_stats["frame_processing_times"]:
        benchmark_stats["avg_processing_time_ms"] = \
            sum(benchmark_stats["frame_processing_times"]) / len(benchmark_stats["frame_processing_times"])
    
    # Calculate duration
    if benchmark_stats["start_time"] and benchmark_stats["end_time"]:
        start = datetime.fromisoformat(benchmark_stats["start_time"])
        end = datetime.fromisoformat(benchmark_stats["end_time"])
        benchmark_stats["total_duration_seconds"] = (end - start).total_seconds()
    
    # Save to file
    with open(BENCHMARK_LOG, 'w') as f:
        json.dump(benchmark_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS SAVED: {BENCHMARK_LOG}")
    print(f"{'='*60}")
    print(f"Total Frames Processed: {benchmark_stats['total_frames_processed']}")
    print(f"Frames with Detections: {benchmark_stats['frames_with_detections']}")
    print(f"Total Tags Detected: {benchmark_stats['total_tags_detected']}")
    print(f"Unique Tag IDs: {benchmark_stats['unique_tag_ids']}")
    print(f"Avg Processing Time: {benchmark_stats['avg_processing_time_ms']:.2f} ms/frame")
    if "total_duration_seconds" in benchmark_stats:
        print(f"Total Duration: {benchmark_stats['total_duration_seconds']:.2f} seconds")
    print(f"{'='*60}\n")

# ============================================================================
# MAIN LOOP - VIDEO MODE
# ============================================================================

def process_video():
    """
    Process pre-recorded video file
    """
    global mqtt_client
    
    print(f"[{get_utc_now()}] 🎬 VIDEO BENCHMARK MODE")
    print(f"[{get_utc_now()}] 📹 Video: {VIDEO_PATH}")
    print(f"[{get_utc_now()}] 🎯 Processing Rate: {PROCESS_FPS} FPS")
    
    # Verify video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"[{get_utc_now()}] ❌ Video file not found: {VIDEO_PATH}")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[{get_utc_now()}] ❌ Failed to open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = total_frames / video_fps
    
    print(f"[{get_utc_now()}] 📊 Video Info:")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Video FPS: {video_fps:.2f}")
    print(f"   - Duration: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes)")
    print(f"   - Frames to Process: {int(duration_seconds * PROCESS_FPS)}")
    
    # Initialize MQTT
    mqtt_client = mqtt.Client()
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[{get_utc_now()}] ✅ MQTT Connected")
        else:
            print(f"[{get_utc_now()}] ⚠️  MQTT Connection Failed (rc={rc})")
    
    mqtt_client.on_connect = on_connect
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        time.sleep(1)  # Wait for connection
    except Exception as e:
        print(f"[{get_utc_now()}] ⚠️  MQTT Error: {e} (continuing without MQTT)")
    
    # Start benchmark
    benchmark_stats["start_time"] = get_utc_now()
    
    # Calculate frame skip interval
    frame_skip = int(video_fps / PROCESS_FPS)
    
    frame_count = 0
    processed_count = 0
    
    try:
        print(f"\n[{get_utc_now()}] 🚀 Starting processing...\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"[{get_utc_now()}] ✅ Reached end of video")
                break
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                processed_count += 1
                
                # Detect tags
                detections, annotated = detect_tags(frame)
                
                # Save annotated frame
                if annotated is not None:
                    cv2.imwrite(ANNOTATED_PATH, annotated)
                
                # Publish to MQTT
                if mqtt_client:
                    publish_to_mqtt(detections, frame_count)
                
                # Update stats
                benchmark_stats["total_frames_processed"] = processed_count
                
                # Log progress
                tag_count = len(detections)
                if tag_count > 0:
                    tag_ids = ", ".join([f"#{d['tag_id']}" for d in detections])
                    print(f"[{get_utc_now()}] Frame {frame_count}/{total_frames} - Tags: {tag_ids}")
                else:
                    # Only log every 10th "no tags" frame to avoid spam
                    if processed_count % 10 == 0:
                        print(f"[{get_utc_now()}] Frame {frame_count}/{total_frames} - No tags")
        
        # End benchmark
        benchmark_stats["end_time"] = get_utc_now()
        
        # Save results
        save_benchmark_results()
    
    except KeyboardInterrupt:
        print(f"\n[{get_utc_now()}] ⏹️  Benchmark interrupted")
        benchmark_stats["end_time"] = get_utc_now()
        save_benchmark_results()
    
    except Exception as e:
        print(f"[{get_utc_now()}] ❌ Error: {e}")
    
    finally:
        cap.release()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if BENCHMARK_MODE:
        process_video()
    else:
        print("Switch to RTSP mode not implemented in benchmark script")
        print("Use original apriltag_detector.py for live stream")

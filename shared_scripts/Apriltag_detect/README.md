# CUDA-Enabled AprilTag Detector

## Overview
This is a CUDA-accelerated AprilTag detection system that:
- Captures frames from RTSP camera stream using OpenCV VideoCapture
- Uses GPU for image preprocessing (resize, color conversion)
- Detects AprilTags using OpenCV's built-in detector
- Publishes detection data and system metrics to MQTT
- Saves annotated images to shared volume

## Features
- **CUDA Acceleration**: GPU-accelerated image preprocessing
- **OpenCV Built-in Detector**: No external dependencies like pupil-apriltags
- **Dual MQTT Topics**: Separate topics for detections and system stats
- **Auto-reconnect**: Automatic RTSP stream reconnection on failure
- **Shared Volume**: Images accessible by Node-RED container

## Files
- `config.py` - Configuration settings
- `apriltag_cuda_detector.py` - Main detection script
- `latest_frame.jpg` - Latest raw frame (auto-generated)
- `latest_detected.jpg` - Latest annotated frame (auto-generated)
- `README.md` - This file

## Configuration
Edit `config.py` to customize:
- RTSP URL and connection settings
- Image resolution (default: 1280x720)
- AprilTag family (default: DICT_APRILTAG_36h11)
- MQTT broker and topics
- Detection interval (default: 5 seconds)
- Camera location and site ID

## MQTT Topics

### 1. Detection Data: `apriltag/teja/detections`
```json
{
  "timestamp": "2024-01-08T12:34:56.789Z",
  "image_path": "/workspace/scripts/Apriltag_detect/latest_detected.jpg",
  "detections": [
    {
      "tag_id": 2,
      "center": [640, 360],
      "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "camera_location": "aadvca.0011.RouterWR255",
      "site_id": "017"
    }
  ],
  "count": 1
}
```

### 2. System Stats: `apriltag/teja/system_stats`
```json
{
  "timestamp": "2024-01-08T12:34:56.789Z",
  "opencv_version": "4.8.1",
  "cuda_available": true,
  "cuda_devices": 1,
  "processing_time_ms": 45.6,
  "fps": 21.9,
  "detections_count": 1,
  "gpu_utilization_percent": 45.2,
  "gpu_temperature_c": 52.3,
  "ram_used_mb": 4096,
  "ram_total_mb": 16384
}
```

## Running the Detector

### Option 1: Run from Container CLI (Current)
```bash
# Enter the opencv-gpu container
sudo docker exec -it opencv-gpu bash

# Navigate to the script directory
cd /workspace/scripts/Apriltag_detect

# Run the detector
python3 apriltag_cuda_detector.py
```

### Option 2: Run in Background
```bash
# Run in detached mode
sudo docker exec -d opencv-gpu python3 /workspace/scripts/Apriltag_detect/apriltag_cuda_detector.py

# View logs
sudo docker logs -f opencv-gpu
```

### Option 3: Update Docker Compose (Future)
Edit `docker-compose.yml`:
```yaml
opencv:
  command: ["python3", "/workspace/scripts/Apriltag_detect/apriltag_cuda_detector.py"]
```

## Stopping the Detector
```bash
# If running in foreground: Ctrl+C

# If running in background:
sudo docker exec opencv-gpu pkill -f apriltag_cuda_detector.py
```

## Troubleshooting

### Check CUDA Availability
```bash
sudo docker exec opencv-gpu python3 -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

### Check RTSP Connection
```bash
# Test RTSP stream
ffmpeg -rtsp_transport tcp -i "rtsps://10.45.41.253:7441/J4ZIcw3NhpITdOkM?enableSrtp" -frames:v 1 test.jpg
```

### Check MQTT Connection
```bash
# Subscribe to detection topic
mosquitto_sub -h aadvncec0039959.ashleyfurniture.com -t "apriltag/teja/detections" -v

# Subscribe to stats topic
mosquitto_sub -h aadvncec0039959.ashleyfurniture.com -t "apriltag/teja/system_stats" -v
```

### View Logs
```bash
# Real-time logs
sudo docker logs -f opencv-gpu

# Last 100 lines
sudo docker logs --tail 100 opencv-gpu
```

## Performance Expectations
- **With CUDA**: ~20-50ms per frame (~20-50 FPS)
- **Without CUDA**: ~50-150ms per frame (~6-20 FPS)
- **Speedup**: 2-3x on preprocessing

## Notes
- OpenCV's AprilTag detector may be less accurate than pupil-apriltags
- CUDA only accelerates preprocessing (resize, color conversion)
- AprilTag detection itself runs on CPU
- Images are saved to shared volume accessible by Node-RED at `/scripts/Apriltag_detect/`


#!/bin/bash
# Diagnostic script for AprilTag detector

echo "=========================================="
echo "AprilTag Detector Diagnostics"
echo "=========================================="
echo ""

echo "1. Checking if detector is running..."
DETECTOR_PID=$(ps aux | grep "apriltag_cuda_detector.py" | grep -v grep | awk '{print $2}')
if [ -z "$DETECTOR_PID" ]; then
    echo "   ❌ Detector is NOT running"
else
    echo "   ✅ Detector is running (PID: $DETECTOR_PID)"
fi
echo ""

echo "2. Checking if MJPEG streamer is running..."
STREAMER_PID=$(ps aux | grep "mjpeg_streamer.py" | grep -v grep | awk '{print $2}')
if [ -z "$STREAMER_PID" ]; then
    echo "   ❌ MJPEG streamer is NOT running"
else
    echo "   ✅ MJPEG streamer is running (PID: $STREAMER_PID)"
fi
echo ""

echo "3. Checking image files..."
if [ -f "/workspace/scripts/Apriltag_detect/latest_detected.jpg" ]; then
    DETECTED_AGE=$(stat -c %Y /workspace/scripts/Apriltag_detect/latest_detected.jpg)
    CURRENT_TIME=$(date +%s)
    AGE_SECONDS=$((CURRENT_TIME - DETECTED_AGE))
    echo "   ✅ latest_detected.jpg exists"
    echo "      Age: ${AGE_SECONDS} seconds old"
    echo "      Modified: $(stat -c %y /workspace/scripts/Apriltag_detect/latest_detected.jpg)"
    
    if [ $AGE_SECONDS -gt 60 ]; then
        echo "      ⚠️  WARNING: Image is more than 1 minute old - detector may not be running"
    fi
else
    echo "   ❌ latest_detected.jpg NOT found"
fi

if [ -f "/workspace/scripts/Apriltag_detect/latest_frame.jpg" ]; then
    echo "   ✅ latest_frame.jpg exists"
else
    echo "   ❌ latest_frame.jpg NOT found"
fi
echo ""

echo "4. Checking RTSP configuration..."
if [ -f "/workspace/scripts/Apriltag_detect/config.py" ]; then
    RTSP_URL=$(grep "RTSP_URL" /workspace/scripts/Apriltag_detect/config.py | head -1)
    echo "   ✅ Config file exists"
    echo "      $RTSP_URL"
else
    echo "   ❌ Config file NOT found"
fi
echo ""

echo "5. Checking AprilTag family configuration..."
if [ -f "/workspace/scripts/Apriltag_detect/config.py" ]; then
    APRILTAG_FAMILY=$(grep "APRILTAG_FAMILY" /workspace/scripts/Apriltag_detect/config.py | head -1)
    echo "   $APRILTAG_FAMILY"
fi
echo ""

echo "6. Checking MQTT configuration..."
if [ -f "/workspace/scripts/Apriltag_detect/config.py" ]; then
    MQTT_BROKER=$(grep "MQTT_BROKER" /workspace/scripts/Apriltag_detect/config.py | head -1)
    echo "   $MQTT_BROKER"
fi
echo ""

echo "7. Checking network connectivity..."
ping -c 1 -W 2 10.45.41.253 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Can reach camera IP (10.45.41.253)"
else
    echo "   ❌ Cannot reach camera IP (10.45.41.253)"
fi

ping -c 1 -W 2 aadvncec0039959.ashleyfurniture.com > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Can reach MQTT broker"
else
    echo "   ❌ Cannot reach MQTT broker"
fi
echo ""

echo "8. Checking CUDA availability..."
python3 -c "import cv2; print('   ✅ CUDA available' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else '   ❌ CUDA not available')" 2>/dev/null
echo ""

echo "9. Checking OpenCV version..."
python3 -c "import cv2; print(f'   OpenCV version: {cv2.__version__}')" 2>/dev/null
echo ""

echo "10. Recent detector logs (last 20 lines)..."
if [ ! -z "$DETECTOR_PID" ]; then
    echo "   (Check terminal where detector is running)"
else
    echo "   Detector not running - no logs available"
fi
echo ""

echo "=========================================="
echo "Diagnostics Complete"
echo "=========================================="
echo ""
echo "Quick Actions:"
echo ""
echo "Start detector:"
echo "  python3 /workspace/scripts/Apriltag_detect/apriltag_cuda_detector.py"
echo ""
echo "Start MJPEG streamer:"
echo "  python3 /workspace/scripts/Apriltag_detect/mjpeg_streamer.py"
echo ""
echo "Start both:"
echo "  bash /workspace/scripts/Apriltag_detect/start_detector_with_stream.sh"
echo ""
echo "Test RTSP stream:"
echo "  ffmpeg -rtsp_transport tcp -i \"rtsps://10.45.41.253:7441/J4ZIcw3NhpITdOkM?enableSrtp\" -frames:v 1 /tmp/test.jpg"
echo ""


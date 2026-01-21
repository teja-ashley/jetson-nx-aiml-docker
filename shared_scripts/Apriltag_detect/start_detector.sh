#!/bin/bash
# Quick start script for AprilTag detector

echo "Starting CUDA-Enabled AprilTag Detector..."
echo "Press Ctrl+C to stop"
echo ""

cd /workspace/scripts/Apriltag_detect
python3 apriltag_cuda_detector.py


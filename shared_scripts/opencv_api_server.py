#!/usr/bin/env python3
"""
Flask REST API Server for OpenCV GPU Operations
Runs inside opencv-test container
Exposes endpoints for Node-RED to call
"""

from flask import Flask, jsonify, request
import cv2
import numpy as np
from datetime import datetime
import logging
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gpu_simple():
    """
    Simple GPU test - same logic as test_gpu_simple.py
    Returns dict with test results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "opencv_version": cv2.__version__,
        "cuda_available": False,
        "cuda_devices": 0,
        "test_result": ""
    }
    
    try:
        # Check CUDA availability
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        result["cuda_available"] = cuda_count > 0
        result["cuda_devices"] = cuda_count
        
        if cuda_count > 0:
            # Create small test image
            cpu_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

            # Upload to GPU (OpenCV 4.10.0+ compatible)
            gpu_img = cv2.cuda.GpuMat(cpu_img)

            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_img, (320, 240))

            # Download result
            result_img = gpu_resized.download()

            result["test_result"] = f"GPU processing successful! Resized {cpu_img.shape} to {result_img.shape}"
        else:
            result["test_result"] = "No CUDA devices found"
            result["status"] = "warning"
            
    except Exception as e:
        result["status"] = "error"
        result["test_result"] = f"Error: {str(e)}"
        result["error_details"] = traceback.format_exc()
        logger.error(f"GPU test failed: {e}\n{traceback.format_exc()}")
    
    return result


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "opencv-gpu-api",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/gpu/test', methods=['GET', 'POST'])
def gpu_test():
    """
    Run simple GPU test
    GET or POST /api/gpu/test
    Returns JSON with test results
    """
    logger.info("GPU test endpoint called")
    
    try:
        result = test_gpu_simple()
        status_code = 200 if result["status"] == "success" else 500
        return jsonify(result), status_code
    
    except Exception as e:
        logger.error(f"Unexpected error in gpu_test: {e}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_details": traceback.format_exc()
        }), 500


@app.route('/api/gpu/info', methods=['GET'])
def gpu_info():
    """
    Get GPU and OpenCV information
    GET /api/gpu/info
    """
    logger.info("GPU info endpoint called")
    
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        
        info = {
            "timestamp": datetime.now().isoformat(),
            "opencv_version": cv2.__version__,
            "cuda_available": cuda_count > 0,
            "cuda_devices": cuda_count,
            "build_info": cv2.getBuildInformation().split('\n')[:20]  # First 20 lines
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        "service": "OpenCV GPU API Server",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/api/gpu/test": "Run GPU test (GET or POST)",
            "/api/gpu/info": "Get GPU and OpenCV information"
        },
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    logger.info("Starting OpenCV GPU API Server...")
    logger.info(f"OpenCV Version: {cv2.__version__}")

    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        logger.info(f"CUDA Devices: {cuda_count}")
    except Exception as e:
        logger.warning(f"Could not check CUDA devices: {e}")

    # Run Flask server
    # 0.0.0.0 allows external connections from other containers
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


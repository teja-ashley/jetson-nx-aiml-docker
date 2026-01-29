#!/usr/bin/env python3
"""
GPU Operations Script
Standalone script for GPU testing and information gathering
Designed to be called as subprocess from Flask API
Executes, returns result, and exits - freeing GPU memory
"""

import sys
import json
import cv2
import numpy as np
from datetime import datetime
import gc
import argparse


def cleanup_gpu():
    """
    Explicitly free GPU memory and cleanup resources
    Called automatically on exit
    """
    try:
        # Force garbage collection
        gc.collect()
        
        # Reset CUDA device to free all GPU memory
        cv2.cuda.resetDevice()
        
    except Exception as e:
        # Don't fail on cleanup errors, just log to stderr
        print(f"GPU cleanup warning: {e}", file=sys.stderr)


def get_gpu_info():
    """
    Get GPU and OpenCV information
    Returns dict with GPU details
    """
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        
        info = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "cuda_available": cuda_count > 0,
            "cuda_devices": cuda_count,
            "build_info": cv2.getBuildInformation().split('\n')[:20]  # First 20 lines
        }
        
        return info
    
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


def run_gpu_test():
    """
    Run GPU test - create image, upload to GPU, resize, download
    Returns dict with test results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
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
            result["input_shape"] = list(cpu_img.shape)
            result["output_shape"] = list(result_img.shape)
        else:
            result["test_result"] = "No CUDA devices found"
            result["status"] = "warning"
            
    except Exception as e:
        result["status"] = "error"
        result["test_result"] = f"Error: {str(e)}"
        result["error"] = str(e)
    
    return result


def main():
    """
    Main entry point - parse arguments and execute requested operation
    """
    parser = argparse.ArgumentParser(description='GPU Operations Script')
    parser.add_argument('--info', action='store_true', help='Get GPU information')
    parser.add_argument('--test', action='store_true', help='Run GPU test')
    
    args = parser.parse_args()
    
    result = None
    exit_code = 0
    
    try:
        if args.info:
            result = get_gpu_info()
        elif args.test:
            result = run_gpu_test()
        else:
            result = {
                "status": "error",
                "error": "No operation specified. Use --info or --test"
            }
            exit_code = 1
        
        # Output result as JSON to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # Output error as JSON to stdout
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        print(json.dumps(error_result, indent=2))
        exit_code = 1
    
    finally:
        # Always cleanup GPU resources before exit
        cleanup_gpu()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


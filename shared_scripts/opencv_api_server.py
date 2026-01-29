#!/usr/bin/env python3
"""
Flask REST API Server for OpenCV GPU Operations
Runs inside opencv-test container
Exposes endpoints for Node-RED to call

Lightweight API server - GPU operations are executed as subprocesses
This ensures GPU crashes don't affect the API server
"""

from flask import Flask, jsonify, request
from datetime import datetime
import logging
import subprocess
import json
import os
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to GPU operations script
GPU_OPERATIONS_SCRIPT = '/workspace/scripts/gpu_operations.py'

# Path to PID test service script
PID_TEST_SCRIPT = '/workspace/scripts/pid_test.py'

# Track container startup time
CONTAINER_START_TIME = datetime.now()
CONTAINER_START_TIMESTAMP = time.time()


def run_gpu_operation(operation, timeout=30):
    """
    Run GPU operation as subprocess

    Args:
        operation: 'info' or 'test'
        timeout: Maximum execution time in seconds

    Returns:
        dict: Result from GPU operation
    """
    try:
        # Build command
        cmd = ['python3', GPU_OPERATIONS_SCRIPT, f'--{operation}']

        logger.info(f"Executing GPU operation: {operation}")

        # Run subprocess with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd='/workspace/scripts'
        )

        # Parse JSON output from stdout
        if result.stdout:
            output = json.loads(result.stdout)
            logger.info(f"GPU operation {operation} completed with status: {output.get('status', 'unknown')}")
            return output
        else:
            # No output - return error
            error_msg = result.stderr if result.stderr else "No output from GPU operation"
            logger.error(f"GPU operation {operation} failed: {error_msg}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }

    except subprocess.TimeoutExpired:
        logger.error(f"GPU operation {operation} timed out after {timeout} seconds")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Operation timed out after {timeout} seconds"
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPU operation output: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Failed to parse output: {str(e)}",
            "raw_output": result.stdout if 'result' in locals() else None
        }

    except Exception as e:
        logger.error(f"Unexpected error running GPU operation {operation}: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def run_pid_test_operation(operation, timeout=10):
    """
    Run PID test service operation as subprocess

    Args:
        operation: 'start', 'stop', or 'status'
        timeout: Maximum execution time in seconds

    Returns:
        dict: Result from PID test operation
    """
    try:
        # Build command
        cmd = ['python3', PID_TEST_SCRIPT, f'--{operation}']

        logger.info(f"Executing PID test operation: {operation}")

        # For 'start' operation, use Popen to run in background
        if operation == 'start':
            # Start process in background
            # CRITICAL: Do NOT capture stderr/stdout to avoid pipe buffer deadlock
            # Let output go to Docker logs instead
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                cwd='/workspace/scripts'
            )

            # Return immediately with process info
            logger.info(f"PID test service started in background (PID: {process.pid})")
            return {
                "status": "started",
                "timestamp": datetime.now().isoformat(),
                "message": "PID test service started in background",
                "process_pid": process.pid
            }

        else:
            # For 'stop' and 'status', wait for completion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd='/workspace/scripts'
            )

            # Parse JSON output from stdout
            if result.stdout:
                output = json.loads(result.stdout)
                logger.info(f"PID test operation {operation} completed with status: {output.get('status', 'unknown')}")
                return output
            else:
                # No output - return error
                error_msg = result.stderr if result.stderr else "No output from PID test operation"
                logger.error(f"PID test operation {operation} failed: {error_msg}")
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg
                }

    except subprocess.TimeoutExpired:
        logger.error(f"PID test operation {operation} timed out after {timeout} seconds")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Operation timed out after {timeout} seconds"
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse PID test operation output: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Failed to parse output: {str(e)}",
            "raw_output": result.stdout if 'result' in locals() else None
        }

    except Exception as e:
        logger.error(f"Unexpected error running PID test operation {operation}: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "opencv-gpu-api",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/container/stats', methods=['GET'])
def container_stats():
    """
    Get container statistics without using Docker socket

    Returns:
        - Container uptime (calculated from API server start time)
        - Start timestamp
        - Current timestamp
        - Uptime in seconds
        - Uptime in human-readable format

    Note: This tracks the API server's uptime, which resets when:
    - Container restarts
    - API server crashes and restarts
    - Manual restart of the API server
    """
    current_time = time.time()
    uptime_seconds = current_time - CONTAINER_START_TIMESTAMP

    # Calculate human-readable uptime
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    uptime_human = f"{days}d {hours}h {minutes}m {seconds}s"

    return jsonify({
        "status": "success",
        "container_name": "opencv-gpu",
        "api_server_started_at": CONTAINER_START_TIME.isoformat(),
        "current_time": datetime.now().isoformat(),
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_human": uptime_human,
        "note": "Uptime resets when container or API server restarts"
    }), 200


@app.route('/api/gpu/test', methods=['GET', 'POST'])
def gpu_test():
    """
    Run GPU test via subprocess
    GET or POST /api/gpu/test
    Returns JSON with test results

    The GPU operation runs as a separate process and exits after completion,
    automatically freeing GPU memory.
    """
    logger.info("GPU test endpoint called")

    # Run GPU test operation
    result = run_gpu_operation('test', timeout=30)

    # Return appropriate status code
    status_code = 200 if result.get("status") == "success" else 500
    return jsonify(result), status_code


@app.route('/api/gpu/info', methods=['GET'])
def gpu_info():
    """
    Get GPU and OpenCV information via subprocess
    GET /api/gpu/info

    The GPU operation runs as a separate process and exits after completion,
    automatically freeing GPU memory.
    """
    logger.info("GPU info endpoint called")

    # Run GPU info operation
    result = run_gpu_operation('info', timeout=10)

    # Return appropriate status code
    status_code = 200 if result.get("status") == "success" else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/start', methods=['POST'])
def pidtest_start():
    """
    Start PID test service in background
    POST /api/pidtest/start

    The service runs as a background process and publishes MQTT messages.
    """
    logger.info("PID test start endpoint called")

    # Run PID test start operation (non-blocking)
    result = run_pid_test_operation('start', timeout=5)

    # Return appropriate status code
    status_code = 200 if result.get("status") == "started" else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/stop', methods=['POST'])
def pidtest_stop():
    """
    Stop PID test service
    POST /api/pidtest/stop
    """
    logger.info("PID test stop endpoint called")

    # Run PID test stop operation
    result = run_pid_test_operation('stop', timeout=5)

    # Return appropriate status code
    status_code = 200 if result.get("status") in ["stopped", "warning"] else 500
    return jsonify(result), status_code


@app.route('/api/pidtest/status', methods=['GET'])
def pidtest_status():
    """
    Check PID test service status
    GET /api/pidtest/status
    """
    logger.info("PID test status endpoint called")

    # Run PID test status operation
    result = run_pid_test_operation('status', timeout=5)

    # Always return 200 for status check
    return jsonify(result), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        "service": "OpenCV GPU API Server",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/api/gpu/test": "Run GPU test (GET or POST)",
            "/api/gpu/info": "Get GPU and OpenCV information",
            "/api/pidtest/start": "Start PID test service (POST)",
            "/api/pidtest/stop": "Stop PID test service (POST)",
            "/api/pidtest/status": "Check PID test service status (GET)"
        },
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    logger.info("Starting OpenCV GPU API Server (Lightweight Mode)...")
    logger.info("GPU operations will be executed as subprocesses")

    # Check if GPU operations script exists
    if os.path.exists(GPU_OPERATIONS_SCRIPT):
        logger.info(f"GPU operations script found: {GPU_OPERATIONS_SCRIPT}")
    else:
        logger.warning(f"GPU operations script not found: {GPU_OPERATIONS_SCRIPT}")

    # Run Flask server
    # 0.0.0.0 allows external connections from other containers
    # No GPU initialization here - keeps server lightweight and reliable
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


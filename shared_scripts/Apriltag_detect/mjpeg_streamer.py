#!/usr/bin/env python3
"""
MJPEG Streaming Server for AprilTag Detection
Streams live video directly from RTSP camera
"""

import cv2
import time
import logging
import threading
from flask import Flask, Response
from config import RTSP_URL, RTSP_TRANSPORT, TARGET_WIDTH, TARGET_HEIGHT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
STREAM_PORT = 5002
STREAM_FPS = 15  # Target FPS for streaming
JPEG_QUALITY = 98  # Higher quality for better AprilTag detection (was 95)

app = Flask(__name__)

# Global frame storage with thread lock
current_frame = None  # JPEG bytes for streaming
current_raw_frame = None  # Raw numpy array for snapshot (higher quality)
frame_timestamp = 0  # Unix timestamp when frame was captured
frame_lock = threading.Lock()

# Maximum age of frame before considering it stale (in seconds)
MAX_FRAME_AGE = 2.0


class RTSPCapture:
    """Thread-safe RTSP capture class with improved stability"""

    def __init__(self, rtsp_url, transport="tcp"):
        self.rtsp_url = rtsp_url
        self.transport = transport
        self.cap = None
        self.running = False
        self.thread = None
        self.consecutive_failures = 0
        self.max_failures_before_full_reset = 3

    def start(self):
        """Start the capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("RTSP capture thread started")

    def stop(self):
        """Stop the capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self._release_capture()
        logger.info("RTSP capture thread stopped")

    def _release_capture(self):
        """Safely release capture resources"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"Error releasing capture: {e}")
            self.cap = None

    def _connect(self):
        """Connect to RTSP stream with optimized settings"""
        global current_frame

        # Release any existing connection first
        self._release_capture()

        logger.info(f"Connecting to RTSP: {self.rtsp_url[:50]}...")

        # Set RTSP transport and buffer options
        import os
        # Reduce buffer size for lower latency, use TCP for reliability
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}|"
            "buffer_size;1024000|"
            "max_delay;500000|"
            "stimeout;5000000"
        )

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self.cap is not None:
            # Set capture properties for better stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for fresh frames

        if not self.cap.isOpened():
            logger.error("Failed to open RTSP stream")
            return False

        logger.info("RTSP stream connected successfully")
        self.consecutive_failures = 0
        return True

    def _capture_loop(self):
        """Main capture loop with improved error handling"""
        global current_frame

        reconnect_delay = 2  # Start with shorter delay
        frame_interval = 1.0 / STREAM_FPS
        frames_since_connect = 0

        while self.running:
            # Connect if needed
            if self.cap is None or not self.cap.isOpened():
                if not self._connect():
                    # Exponential backoff on repeated failures
                    self.consecutive_failures += 1
                    delay = min(reconnect_delay * self.consecutive_failures, 30)
                    logger.warning(f"Reconnecting in {delay}s... (attempt {self.consecutive_failures})")
                    time.sleep(delay)
                    continue
                frames_since_connect = 0

            try:
                # Grab and retrieve separately for better error handling
                if not self.cap.grab():
                    logger.warning("Failed to grab frame")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures_before_full_reset:
                        logger.warning("Too many failures, forcing full reconnect")
                        self._release_capture()
                        time.sleep(1)
                    continue

                ret, frame = self.cap.retrieve()

                if not ret or frame is None:
                    logger.warning("Failed to retrieve frame, reconnecting...")
                    self._release_capture()
                    time.sleep(0.5)
                    continue

                # Reset failure counter on successful frame
                self.consecutive_failures = 0
                frames_since_connect += 1

                # Skip first few frames after reconnect (may be stale)
                if frames_since_connect < 3:
                    time.sleep(0.1)
                    continue

                # Resize if needed
                if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                # Encode to JPEG for streaming
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)

                if ret:
                    with frame_lock:
                        global current_raw_frame, frame_timestamp
                        current_frame = jpeg.tobytes()
                        current_raw_frame = frame.copy()  # Store raw frame for snapshot
                        frame_timestamp = time.time()  # Record when frame was captured

                # Control frame rate
                time.sleep(frame_interval)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.5)


# Global RTSP capture instance
rtsp_capture = None


def generate_frames():
    """Generator function to yield MJPEG frames"""
    global current_frame

    while True:
        with frame_lock:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')

        time.sleep(1.0 / STREAM_FPS)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/snapshot')
def snapshot():
    """Return a single high-quality JPEG frame for AprilTag detector"""
    global current_raw_frame, frame_timestamp

    with frame_lock:
        if current_raw_frame is not None:
            frame_age = time.time() - frame_timestamp

            # Reject stale frames - force detector to wait for fresh frame
            if frame_age > MAX_FRAME_AGE:
                logger.warning(f"Frame is stale ({frame_age:.1f}s old), waiting for fresh frame")
                return Response("Frame too old", status=503)

            # Encode with maximum quality for best detection results
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            ret, jpeg = cv2.imencode('.jpg', current_raw_frame, encode_params)
            if ret:
                response = Response(jpeg.tobytes(), mimetype='image/jpeg')
                response.headers['X-Frame-Age'] = str(round(frame_age, 3))
                return response
        return Response("No frame available", status=503)


@app.route('/health')
def health():
    """Health check endpoint"""
    global current_frame, frame_timestamp

    with frame_lock:
        has_frame = current_frame is not None
        frame_age = time.time() - frame_timestamp if frame_timestamp > 0 else -1
        is_fresh = frame_age < MAX_FRAME_AGE if has_frame else False

    return {
        "status": "ok",
        "rtsp_url": RTSP_URL[:50] + "...",
        "has_frame": has_frame,
        "frame_age_seconds": round(frame_age, 2),
        "is_fresh": is_fresh,
        "stream_fps": STREAM_FPS
    }


@app.route('/')
def index():
    """Simple HTML page to view the stream"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live RTSP Stream</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background: #1a1a1a;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }}
            h1 {{ color: #00b500; }}
            img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #00b500;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 181, 0, 0.3);
            }}
            .info {{
                margin-top: 20px;
                color: #999;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <h1>Live RTSP Stream</h1>
        <img src="/video_feed" alt="Live Stream">
        <div class="info">
            <p>MJPEG Stream from RTSP camera</p>
            <p>Target FPS: {STREAM_FPS}</p>
        </div>
    </body>
    </html>
    """


if __name__ == '__main__':
    logger.info(f"Starting MJPEG streamer on port {STREAM_PORT}")
    logger.info(f"RTSP URL: {RTSP_URL[:50]}...")
    logger.info(f"Target FPS: {STREAM_FPS}")

    # Start RTSP capture thread
    rtsp_capture = RTSPCapture(RTSP_URL, RTSP_TRANSPORT)
    rtsp_capture.start()

    # Give capture thread time to connect
    time.sleep(2)

    logger.info(f"Access stream at: http://0.0.0.0:{STREAM_PORT}/video_feed")

    try:
        app.run(
            host='0.0.0.0',
            port=STREAM_PORT,
            debug=False,
            threaded=True
        )
    finally:
        rtsp_capture.stop()


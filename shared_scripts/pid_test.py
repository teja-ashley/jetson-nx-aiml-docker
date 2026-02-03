#!/usr/bin/env python3
"""
PID Test Service
Simple long-running service for testing process management
Sends MQTT messages with timestamps when running
"""

import sys
import json
import time
import signal
import os
from datetime import datetime
import argparse
import paho.mqtt.client as mqtt

# Configuration
MQTT_BROKER = "aadvncec0039959.ashleyfurniture.com"
MQTT_PORT = 1883
MQTT_TOPIC = "/docker/test/pid"
PID_FILE = "/tmp/pid_test.pid"
LOG_FILE = "/tmp/pid_test.log"
PUBLISH_INTERVAL = 0.5  # seconds


def log_message(msg):
    """Write message to both stderr and log file"""
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] {msg}"
    print(log_line, file=sys.stderr)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_line + '\n')
    except Exception as e:
        print(f"Failed to write to log file: {e}", file=sys.stderr)


class PIDTestService:
    """Simple service that publishes timestamps to MQTT - uses same pattern as simple_mqtt_test.py"""

    def __init__(self):
        self.running = True
        self.mqtt_client = None

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        log_message(f"Received signal {signum}, shutting down...")
        self.running = False

    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            log_message("MQTT connected successfully")
        else:
            log_message(f"MQTT connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        log_message(f"MQTT disconnected with code {rc}")

    def on_publish(self, client, userdata, mid):
        """MQTT publish callback - don't log to reduce noise"""
        pass

    def run(self):
        """Main service loop - publishes timestamps to MQTT using paho-mqtt (same pattern as simple_mqtt_test.py)"""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Write PID file
        try:
            with open(PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            return {"status": "error", "error": f"Failed to write PID file: {e}"}

        # Create MQTT client (same as simple_mqtt_test.py)
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_publish = self.on_publish

        # Connect to broker
        log_message(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            log_message(f"Failed to connect to MQTT broker: {e}")
            return {"status": "error", "error": f"MQTT connection failed: {e}"}

        # Wait for connection
        time.sleep(1)

        # Main loop
        message_count = 0
        log_message(f"PID Test Service started (PID: {os.getpid()})")
        log_message(f"Publishing to {MQTT_BROKER}:{MQTT_PORT} topic {MQTT_TOPIC}")

        while self.running:
            try:
                # Create message with timestamp
                message = {
                    "timestamp": datetime.now().isoformat(),
                    "message_count": message_count,
                    "pid": os.getpid(),
                    "status": "running"
                }

                # Publish to MQTT (same as simple_mqtt_test.py)
                self.mqtt_client.publish(MQTT_TOPIC, json.dumps(message), qos=0)
                message_count += 1
                log_message(f"Published message #{message_count}")

                # Sleep between publishes
                time.sleep(PUBLISH_INTERVAL)

            except Exception as e:
                log_message(f"Error in main loop: {e}")
                time.sleep(1)

        # Cleanup
        log_message("Stopping MQTT client...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        time.sleep(1)  # Give time for disconnect to complete

        # Remove PID file
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception as e:
            log_message(f"Failed to remove PID file: {e}")

        log_message(f"PID Test Service stopped. Total messages: {message_count}")
        return {"status": "stopped", "message_count": message_count}


def start_service():
    """Start the service in background"""
    # Check if already running
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is actually running
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                return {
                    "status": "error",
                    "error": f"Service already running with PID {pid}"
                }
            except OSError:
                # Process not running, remove stale PID file
                os.remove(PID_FILE)
        except Exception as e:
            return {"status": "error", "error": f"Error checking PID file: {e}"}
    
    # Start service
    service = PIDTestService()
    result = service.run()
    
    return result


def stop_service():
    """Stop the running service"""
    if not os.path.exists(PID_FILE):
        return {"status": "error", "error": "Service is not running (no PID file found)"}

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        # Send SIGTERM to process
        os.kill(pid, signal.SIGTERM)

        # Wait up to 5 seconds for graceful shutdown (process sleeps for 2 seconds between publishes)
        for i in range(5):
            time.sleep(1)
            try:
                os.kill(pid, 0)
                # Process still running, keep waiting
            except OSError:
                # Process stopped
                return {
                    "status": "stopped",
                    "message": f"Service stopped (PID {pid})"
                }

        # If we get here, process is still running after 5 seconds
        return {
            "status": "warning",
            "message": f"SIGTERM sent to PID {pid}, but process still running after 5 seconds"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_status():
    """Check if service is running"""
    if not os.path.exists(PID_FILE):
        return {
            "status": "stopped",
            "running": False,
            "message": "Service is not running"
        }
    
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is running
        try:
            os.kill(pid, 0)
            return {
                "status": "running",
                "running": True,
                "pid": pid,
                "message": f"Service is running with PID {pid}"
            }
        except OSError:
            # PID file exists but process not running
            os.remove(PID_FILE)
            return {
                "status": "stopped",
                "running": False,
                "message": "Service stopped (stale PID file removed)"
            }
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='PID Test Service')
    parser.add_argument('--start', action='store_true', help='Start the service')
    parser.add_argument('--stop', action='store_true', help='Stop the service')
    parser.add_argument('--status', action='store_true', help='Check service status')
    
    args = parser.parse_args()
    
    result = None
    
    if args.start:
        result = start_service()
    elif args.stop:
        result = stop_service()
    elif args.status:
        result = check_status()
    else:
        result = {
            "status": "error",
            "error": "No operation specified. Use --start, --stop, or --status"
        }
    
    # Output result as JSON
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("status") in ["running", "stopped", "started"] else 1)


if __name__ == "__main__":
    main()


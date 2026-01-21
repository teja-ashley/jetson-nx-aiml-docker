#!/bin/bash
# Start AprilTag detector with MJPEG streaming
# Robust version with auto-restart, logging, and signal handling
# to run this script- 
# sudo docker exec -d opencv-gpu /bin/bash -c "nohup /workspace/scripts/Apriltag_detect/start_detector_with_stream.sh > /workspace/scripts/Apriltag_detect/logs/startup.log 2>&1 &"
# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR="/workspace/scripts/Apriltag_detect/logs"
LOG_FILE="${LOG_DIR}/detector_$(date +%Y%m%d).log"
PID_FILE="/tmp/apriltag_detector.pid"
STREAMER_PID_FILE="/tmp/mjpeg_streamer.pid"
MAX_RESTART_DELAY=300  # Maximum restart delay in seconds (5 minutes)
MIN_RESTART_DELAY=5    # Minimum restart delay in seconds
RESTART_DELAY=$MIN_RESTART_DELAY
UPTIME_THRESHOLD=60    # If running longer than this, reset restart delay

# ============================================================================
# SETUP
# ============================================================================
mkdir -p "$LOG_DIR"

# Logging function with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================================================"
log "Starting AprilTag CUDA Detector with MJPEG Streaming"
log "========================================================================"
log "PID: $$"
log "Log file: $LOG_FILE"

# Save our PID
echo $$ > "$PID_FILE"

# ============================================================================
# SIGNAL HANDLING
# ============================================================================
# Track if we should exit completely (only on explicit SIGINT/SIGTERM)
SHUTDOWN_REQUESTED=false
STREAMER_PID=""

cleanup() {
    log "Cleanup requested (signal received)"
    SHUTDOWN_REQUESTED=true

    if [ -n "$STREAMER_PID" ] && kill -0 $STREAMER_PID 2>/dev/null; then
        log "Stopping MJPEG streamer (PID: $STREAMER_PID)..."
        kill $STREAMER_PID 2>/dev/null
        wait $STREAMER_PID 2>/dev/null
    fi

    rm -f "$PID_FILE" "$STREAMER_PID_FILE"
    log "Cleanup complete. Exiting."
    exit 0
}

# Handle SIGINT (Ctrl+C) and SIGTERM - these request full shutdown
trap cleanup SIGINT SIGTERM

# Ignore SIGHUP - this prevents SSH disconnection from killing the script
trap '' SIGHUP

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
start_streamer() {
    log "Starting MJPEG streamer on port 5002..."
    nohup python3 /workspace/scripts/Apriltag_detect/mjpeg_streamer.py >> "$LOG_FILE" 2>&1 &
    STREAMER_PID=$!
    echo $STREAMER_PID > "$STREAMER_PID_FILE"
    log "MJPEG streamer started with PID: $STREAMER_PID"
    sleep 3
}

check_streamer() {
    if [ -n "$STREAMER_PID" ] && kill -0 $STREAMER_PID 2>/dev/null; then
        return 0  # Running
    else
        log "MJPEG streamer not running, restarting..."
        start_streamer
        return 1  # Was restarted
    fi
}

# ============================================================================
# MAIN LOOP
# ============================================================================
start_streamer

log "Starting AprilTag detector (with auto-restart on ANY exit)..."
RESTART_COUNT=0

while true; do
    # Check if shutdown was requested
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        log "Shutdown requested, exiting main loop"
        break
    fi

    # Ensure streamer is running before starting detector
    check_streamer

    START_TIME=$(date +%s)
    RESTART_COUNT=$((RESTART_COUNT + 1))
    log "Starting detector (restart #$RESTART_COUNT)..."

    # Run detector and capture exit code
    python3 /workspace/scripts/Apriltag_detect/apriltag_cuda_detector.py 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    END_TIME=$(date +%s)
    UPTIME=$((END_TIME - START_TIME))

    log "Detector exited with code $EXIT_CODE after ${UPTIME}s"

    # Check if shutdown was requested while detector was running
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        log "Shutdown requested during detector run, exiting"
        break
    fi

    # Adjust restart delay based on uptime (exponential backoff with reset)
    if [ $UPTIME -ge $UPTIME_THRESHOLD ]; then
        # Ran for a while, reset delay
        RESTART_DELAY=$MIN_RESTART_DELAY
        log "Detector ran for ${UPTIME}s, resetting restart delay to ${RESTART_DELAY}s"
    else
        # Quick crash, increase delay (exponential backoff)
        RESTART_DELAY=$((RESTART_DELAY * 2))
        if [ $RESTART_DELAY -gt $MAX_RESTART_DELAY ]; then
            RESTART_DELAY=$MAX_RESTART_DELAY
        fi
        log "Quick exit detected, increasing restart delay to ${RESTART_DELAY}s"
    fi

    # Check and restart streamer if needed
    check_streamer

    log "Restarting detector in ${RESTART_DELAY}s..."
    sleep $RESTART_DELAY
done

# Final cleanup
cleanup

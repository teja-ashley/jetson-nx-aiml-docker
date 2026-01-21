# Jetson AGX AI/ML Docker Setup

Docker-based setup for running computer vision and AprilTag detection on NVIDIA Jetson devices.

## Repository Structure

```
.
├── Docker_files/           # Docker configuration files
│   ├── docker-compose.yml
│   ├── Dockerfile.opencv
│   ├── requirements.txt
│   └── pupil_apriltags-*.whl
└── shared_scripts/         # Python scripts for computer vision tasks
    ├── opencv_api_server.py
    ├── apriltag_detector_benchmark.py
    └── ... (other scripts)
```

## Prerequisites

- NVIDIA Jetson device (tested on AGX Orin Developer Kit)
- JetPack 6.x installed
- Docker and Docker Compose installed
- NVIDIA Container Runtime configured

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/teja-gitcode/jetson-nx-aiml-docker.git
   cd jetson-nx-aiml-docker
   ```

2. Navigate to Docker_files directory:
   ```bash
   cd Docker_files
   ```

3. Build and start the containers:
   ```bash
   sudo docker compose build
   sudo docker compose up -d
   ```

4. Check container status:
   ```bash
   sudo docker compose ps
   ```

## Services

### OpenCV GPU Container
- **Port**: 5001 (API), 5002 (MJPEG streaming)
- **Purpose**: GPU-accelerated OpenCV operations and AprilTag detection
- **Base Image**: dustynv/l4t-pytorch:r36.4.0

### Node-RED Container
- **Port**: 1881
- **Purpose**: Flow-based programming for automation and monitoring
- **Data Volume**: `/opt/cartloading/nodered/data`

## Configuration

The `shared_scripts/` directory is mounted as a volume in the opencv container at `/workspace/scripts/`.
Any changes to Python scripts are immediately available inside the container.

## Health Checks

Both containers have health checks configured:
- OpenCV container: HTTP check on `/health` endpoint
- Node-RED container: Custom health check script

## Resource Limits

- Node-RED: 512MB memory limit, 256MB reserved
- OpenCV: GPU access with all NVIDIA capabilities

## Logging

Logs are configured with:
- Max size: 50MB (Node-RED), 100MB (OpenCV)
- Max files: 5
- Compression: enabled

## Network

Both containers run on a custom bridge network: `cartloading-network`

## Troubleshooting

### Container won't start
Check logs:
```bash
sudo docker compose logs opencv
sudo docker compose logs nodered
```

### GPU not accessible
Verify NVIDIA runtime:
```bash
sudo docker run --rm --runtime=nvidia dustynv/l4t-pytorch:r36.4.0 nvidia-smi
```

### Disk space issues
Clean up Docker:
```bash
sudo docker system prune -a --volumes -f
sudo docker builder prune -a -f
```


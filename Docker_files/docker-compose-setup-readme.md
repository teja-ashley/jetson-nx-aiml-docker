# Docker Compose Setup Guide for Node-RED + OpenCV GPU Containers


### Step 1: Stop and Remove Existing Containers

```bash
# Stop containers
docker stop cartloading-nodered opencv-test

# Remove containers (data in /opt/cartloading/nodered/data is preserved)
docker rm cartloading-nodered opencv-test
```

### Step 2: Create Docker Compose File

Create file `/home/afi/Documents/Teja/docker-compose.yml`:

```yaml
services:
  nodered:
    container_name: nodered
    image: nodered/node-red:latest
    restart: unless-stopped
    ports:
      - "1881:1880"
    volumes:
      - /opt/cartloading/nodered/data:/data
      - /home/afi/Documents/Teja/shared_scripts:/scripts:ro
      - /var/run/docker.sock:/var/run/docker.sock # for monitoring containers from node-red (optional)
    networks:
      - cartloading-network
    environment:
      - TZ=America/New_York
    healthcheck:
      test: ["CMD-SHELL", "node /healthcheck.js"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
        compress: "true"
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  opencv:
    container_name: opencv-gpu
    #image: dustynv/l4t-pytorch:r36.2.0 # added image in Dockerfile.opencv
    build:
      context: .                     #file in same directory as docker-compose.yml file path. for now /Documents/Teja
      dockerfile: Dockerfile.opencv 
    restart: unless-stopped
    runtime: nvidia
    volumes:
      - /home/afi/Documents/Teja/shared_scripts:/workspace/scripts
    networks:
      - cartloading-network
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - TZ=America/New_York
    command: ["python3", "/workspace/scripts/opencv_api_server.py"]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:5001/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        compress: "true"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  cartloading-network:
    driver: bridge
    name: cartloading-network
```
# read this before proceeding to next steps 
Getting error with flask and other modules not found. 
Added Dockerfile.opencv file and requirements.txt file for building and installing flask and other modules.
# Dockerfile.opencv
```
FROM dustynv/l4t-pytorch:r36.2.0
COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://pypi.org/simple/ --ignore-installed -r /tmp/requirements.txt
```

# add required pip modules in requirements.txt file. 

# need to modify the docker-compose.yml file to build the image from Dockerfile.opencv if not done earlier
``` opencv:
    container_name: opencv-gpu
    #image: dustynv/l4t-pytorch:r36.2.0 # added image in Dockerfile.opencv
    build:
      context: .                     #file in same directory as docker-compose.yml file path. for now /Documents/Teja
      dockerfile: Dockerfile.opencv 
    #rest of code from restart 
  ```

### Step 3: Validate and Start Services

```bash
# Navigate to project directory
cd /home/afi/Documents/Teja

# Validate docker-compose file
docker compose config

# Build images when Dockerfile.opencv and requirements.txt for pip modules are used 
docker compose build opencv --no-cache

# Start services in detached mode
docker compose up -d

# Verify containers are running
docker compose ps

# Check logs
docker compose logs -f

# delete image cache After building
sudo docker builder prune -a -f
sudo docker buildx prune -a -f
sudo docker image prune -a -f

```

### Step 4: Test Internal Network Communication

From Node-RED, you can now call the OpenCV API using the container name:

```bash

# Test internal network (exec into nodered container)
docker exec nodered wget -qO- http://opencv-gpu:5001/health
```

## Production Features Explained

###  Policies added 

 `unless-stopped`- Always restart unless manually stopped. Survives host reboots. |

### Logging Configuration
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "50m"    # Max size per log file
    max-file: "5"      # Keep 5 rotated files
    compress: "true"   # Compress rotated logs
```



##  Commands for reference

```bash
# View real-time logs
docker compose logs -f

# View logs for specific service
docker compose logs -f opencv

# Restart a service
docker compose restart nodered

# Stop all services
docker compose down

# Stop and remove volumes (deletesall  data)
docker compose down -v

# Update images and recreate
docker compose pull
docker compose up -d

# Check container health
docker inspect --format='{{.State.Health.Status}}' nodered

# View resource usage
docker stats
```

---

## Monitoring Commands

```bash
# Container health status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Resource monitoring
docker stats --no-stream

# View logs with timestamps
docker compose logs -t --since 1h

# Check restart count
docker inspect --format='{{.RestartCount}}' opencv-gpu
```

---

## Troubleshooting

### OpenCV container not starting
```bash
# Check logs
docker compose logs opencv

# Verify nvidia runtime
docker info | grep -i runtime

# Test GPU access manually
docker run --rm --runtime=nvidia dustynv/l4t-pytorch:r36.2.0 nvidia-smi
```

### Network connectivity issues
```bash
# Verify network exists
docker network ls | grep cartloading

# Inspect network
docker network inspect cartloading-network

# Test DNS resolution
docker exec nodered nslookup opencv-gpu
```

## File Locations

 Docker Compose file -  `/home/afi/Documents/Teja/docker-compose.yml` 

 Node-RED flows -  `/opt/cartloading/nodered/data/flows.json` 
 
 Shared scripts -  `/home/afi/Documents/Teja/shared_scripts/` 
 
 Container logs -  `/var/lib/docker/containers/<id>/*.log` 


## API Endpoints (OpenCV Container)


| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://opencv-gpu:5001/health` | GET | Health check |
| `http://opencv-gpu:5001/api/gpu/test` | GET/POST | Run GPU test |
| `http://opencv-gpu:5001/api/gpu/info` | GET | Get GPU info |

## Other commands 
 tegrastats
 
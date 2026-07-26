# Deployment Guide

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- GPU support (optional, for faster inference)

### Quick Start

```bash
# Clone repository
git clone <repo-url>
cd sentinel-stream

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Services

1. **Redis**: Port 6379
2. **Inference API**: Port 8000
3. **Dashboard**: Port 3000

### Configuration

Edit `docker-compose.yml` to customize:
- Port mappings
- Resource limits
- Environment variables

## Manual Deployment

### 1. Install Dependencies

```bash
# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Start Redis

```bash
redis-server
```

### 3. Train Model (if not using pre-trained)

```bash
# Prepare dataset
python scripts/prepare_dataset.py --dataset CIC-IDS2017

# Train model
python scripts/train_model.py --config configs/tgn_config.yaml
```

### 4. Start Inference API

```bash
python src/inference/api.py
```

### 5. Start Dashboard

```bash
cd dashboard
npm install
npm start
```

### 6. Start Packet Capture

```bash
# Requires root/sudo
sudo python scripts/capture_live.py --interface eth0
```

## Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests (to be created).

## Production Considerations

1. **Security**
   - Use CA-signed certificates for OpenSSL
   - Enable Redis AUTH
   - Use secrets management (e.g., Kubernetes secrets)
   - Implement rate limiting on API

2. **Monitoring**
   - Add Prometheus metrics
   - Set up Grafana dashboards
   - Monitor Redis memory usage
   - Track model inference latency

3. **High Availability**
   - Deploy multiple inference API instances
   - Use Redis Sentinel for Redis HA
   - Implement health checks
   - Set up auto-scaling

4. **Performance**
   - Use GPU for model inference
   - Tune Redis memory limits
   - Optimize batch sizes
   - Use connection pooling


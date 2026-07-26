# Quick Start Guide

Get Sentinel-Stream up and running in minutes!

## Prerequisites

- Python 3.9+
- Node.js 16+ (for dashboard)
- Redis
- libpcap (for packet capture)
- Conda (optional, but recommended)

## Installation

### Option 1: Conda (Recommended for Development)

```bash
# First, install system dependencies (libpcap-dev)
# Ubuntu/Debian:
sudo apt-get install libpcap-dev
# macOS:
brew install libpcap
# Or use the automated script:
bash scripts/install_system_deps.sh

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate sentinel-stream
```

### Option 2: Docker (Recommended for Deployment)

```bash
# Clone repository
git clone <repo-url>
cd sentinel-stream

# Start all services
docker-compose up -d

# Access dashboard at http://localhost:3000
# API at http://localhost:8000
```

### Option 3: Manual Setup with pip

```bash
# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or use Makefile
make setup

# Or install directly
pip install -r requirements.txt
```

## Training a Model

```bash
# Prepare dataset (downloads CIC-IDS2017)
python scripts/prepare_dataset.py --dataset CIC-IDS2017

# Train model
python scripts/train_model.py --config configs/tgn_config.yaml

# Or use Makefile
make train
```

## Running the System

### 1. Start Redis

```bash
redis-server
```

### 2. Start Inference API

```bash
python src/inference/api.py
```

Or with uvicorn directly:
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### 3. Start Dashboard

```bash
cd dashboard
npm install
npm start
```

Dashboard will be available at http://localhost:3000

### 4. Start Packet Capture

```bash
# Requires root/sudo for packet capture
sudo python scripts/capture_live.py --interface eth0
```

Replace `eth0` with your network interface name.

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "src_ip": "192.168.1.1",
      "dst_ip": "192.168.1.2",
      "src_port": 12345,
      "dst_port": 80,
      "protocol": 6,
      "timestamp": 1234567890.0,
      "packet_size": 1500
    }
  ]'
```

## Configuration

Edit configuration files in `configs/`:
- `tgn_config.yaml` - Model and training settings
- `inference_config.yaml` - Inference service settings

## Troubleshooting

### Redis Connection Error
- Ensure Redis is running: `redis-cli ping`
- Check Redis host/port in config files

### Packet Capture Fails
- Install libpcap: `sudo apt-get install libpcap-dev` (Ubuntu/Debian)
- Run with sudo/root privileges
- Check interface name: `ip addr` or `ifconfig`

### Model Not Found
- Train model first: `make train`
- Check model path in `configs/inference_config.yaml`

## Next Steps

- Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Read [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment
- Check [README.md](README.md) for full documentation


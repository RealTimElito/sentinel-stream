# Sentinel-Stream: Real-Time Network Anomaly Detection System

## Overview

Sentinel-Stream is an advanced network security monitoring system that leverages Temporal Graph Networks (TGN) to detect anomalies and attacks in real-time network traffic. The system combines graph neural networks, secure data ingestion, and a modern web dashboard to provide comprehensive network threat detection.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Packet Capture │ --> │ Redis Buffer │ --> │ ML Inference│ --> │   Dashboard  │
│   (libpcap)     │     │   (Streams)  │     │   (FastAPI) │     │   (React)    │
└─────────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

## Features

- **Real-time Packet Capture**: Secure network packet capture using libpcap
- **Graph-based ML**: Temporal Graph Networks for anomaly detection
- **Explainable AI**: SHAP integration for attack attribution
- **Live Dashboard**: Interactive network topology visualization
- **Secure Transmission**: OpenSSL-encrypted data pipeline
- **Production Ready**: Docker/Kubernetes deployment support

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Redis
- libpcap development libraries

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd sentinel-stream

# Install system dependencies (libpcap-dev)
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

# Start services with Docker Compose
docker-compose up -d
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repo-url>
cd sentinel-stream

# Install Python dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Start services with Docker Compose
docker-compose up -d
```

### Training the Model

```bash
# Download and preprocess dataset (automatic download via Kaggle)
python scripts/prepare_dataset.py --dataset CIC-IDS2017 --download

# Or download separately
python scripts/download_dataset.py --method kaggle

# Then preprocess
python scripts/prepare_dataset.py --dataset CIC-IDS2017

# Train the TGN model
python scripts/train_model.py --config configs/tgn_config.yaml
```

### Running the System

```bash
# Start packet capture (requires root/sudo)
sudo python src/ingestion/capture.py --interface eth0

# Start inference service
python src/inference/api.py

# Start dashboard (in separate terminal)
cd dashboard && npm install && npm start
```

## Project Structure

```
sentinel-stream/
├── src/
│   ├── ingestion/          # Phase 1: Packet capture & ingestion
│   ├── ml/                 # Phase 2: ML models & training
│   ├── inference/          # Phase 3: Inference engine
│   └── utils/              # Shared utilities
├── dashboard/              # Phase 3: React frontend
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Development Roadmap

### Phase 1: Data Engineering & Foundation ✅
- [x] Dataset selection (CIC-IDS2017/CSE-CIC-IDS2018)
- [x] Feature engineering pipeline
- [x] Secure packet capture with libpcap
- [x] Redis buffer implementation
- [x] Secure transmission layer (socat/OpenSSL)

### Phase 2: ML Core ✅
- [x] Graph construction from network flows
- [x] Temporal Graph Network (TGN) implementation
- [x] Self-supervised pre-training
- [x] Supervised fine-tuning
- [x] SHAP explainability integration

### Phase 3: Systems & Real-time Integration ✅
- [x] FastAPI inference engine
- [x] Redis Streams processing
- [x] React dashboard with topology visualization

### Phase 4: DevOps & Polish ✅
- [x] Docker Compose deployment
- [x] CI/CD pipeline (GitHub Actions)
- [x] Comprehensive documentation

## License

MIT License

## Citation

If you use Sentinel-Stream in your research, please cite:

```bibtex
@software{sentinel-stream,
  title={Sentinel-Stream: Real-Time Network Anomaly Detection},
  author={RealTimElito},
  year={2024},
  url={https://github.com/RealTimElito/sentinel-stream}
}
```


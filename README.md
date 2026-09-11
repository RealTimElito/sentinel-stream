# Sentinel-Stream: Real-Time Network Anomaly Detection

Sentinel-Stream detects anomalous network flows with a Temporal Graph Network (TGN), Redis buffering, a FastAPI inference service, and a React dashboard.

> **Status:** v0.1.0 beta — core train/infer paths work; train your own checkpoint before production use.

## Architecture

```
Packet Capture --> Redis Streams --> TGN Inference (FastAPI) --> React Dashboard
```

## Features

- Packet capture with scapy (optional `pypcap` if libpcap is installed)
- Flow feature engineering and temporal graph construction
- TGN model with node memory, temporal encoding, and TransformerConv layers
- FastAPI `/predict`, `/demo/predict`, `/health`, and WebSocket endpoints
- React topology view driven by API responses
- Docker Compose + GitHub Actions CI

## Requirements

- Python 3.9+
- Redis (for live streaming; optional for `/predict`)
- Node.js 18+ (dashboard)
- Optional: Docker, CUDA, libpcap-dev

## Quick start

```bash
# System deps (Ubuntu/Debian)
bash scripts/install_system_deps.sh

# Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate sentinel-stream

# Demo model + API
python scripts/create_demo_model.py
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# Dashboard
cd dashboard && npm install && npm start
```

Smoke-check the API:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/demo/predict | jq .
```

## Train on CIC-IDS2017

Place MachineLearningCSV files under `data/raw/`, then:

```bash
python scripts/prepare_dataset.py --dataset CIC-IDS2017
python scripts/train_model.py --config configs/tgn_config.yaml --max-graphs 64
```

## Live capture

```bash
docker compose up -d redis
sudo python scripts/capture_live.py --interface eth0
# inference service consumes Redis stream when redis.enabled=true
```

## Tests & lint

```bash
make test
make lint
```

## Project layout

```
src/ingestion/    # capture + Redis buffer
src/ml/           # TGN, training, explainability
src/inference/    # FastAPI service
src/utils/        # feature engineering
dashboard/        # React UI
scripts/          # dataset, train, demo model
tests/            # pytest suite
```

## License

MIT

## Citation

```bibtex
@software{sentinel-stream,
  title={Sentinel-Stream: Real-Time Network Anomaly Detection},
  author={RealTimElito},
  year={2026},
  url={https://github.com/RealTimElito/sentinel-stream}
}
```

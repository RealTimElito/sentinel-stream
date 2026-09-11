# Quick Start Guide

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional packet capture extras:

```bash
bash scripts/install_system_deps.sh
pip install '.[pcap]'
```

## 2. Demo checkpoint + API

```bash
python scripts/create_demo_model.py
PYTHONPATH=. uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/demo/predict
```

## 3. Dashboard

```bash
cd dashboard
npm install
npm start
```

Open http://localhost:3000 — the UI polls `/demo/predict` and renders the returned topology.

## 4. Train on CIC-IDS2017

Put CSV files in `data/raw/`, then:

```bash
python scripts/prepare_dataset.py --dataset CIC-IDS2017
python scripts/train_model.py --max-graphs 64
```

## 5. Docker

```bash
docker compose up -d --build
```

Redis is required for live capture streaming. The `/predict` API works without Redis.

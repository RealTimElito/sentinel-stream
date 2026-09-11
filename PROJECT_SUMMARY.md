# Sentinel-Stream Project Summary

## Status

**v0.1.0 beta** — core train / infer / dashboard paths are working and covered by tests.

## What works

- Feature engineering and temporal graph construction
- TGN forward pass (memory + temporal encoding + TransformerConv)
- Dataset mapping for CIC-IDS2017 MachineLearningCSV exports
- FastAPI `/health`, `/predict`, `/demo/predict`, WebSocket
- React dashboard consuming API topology responses
- Demo checkpoint via `scripts/create_demo_model.py`
- Docker Compose + GitHub Actions CI (pytest + black/flake8)

## Caveats

- Train on real data before production use; demo weights are synthetic
- Live packet capture needs privileges + scapy/pypcap
- Redis is optional for `/predict`, required for stream ingestion
- Explainability uses occlusion / score magnitude (not full SHAP KernelExplainer)

## Quick verification

```bash
pip install -r requirements.txt
python scripts/create_demo_model.py
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. uvicorn src.inference.api:app --port 8000
```

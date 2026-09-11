# Changelog

All notable changes to Sentinel-Stream will be documented in this file.

## [0.1.0] - 2026-09-11

### Added
- Initial public release of Sentinel-Stream
- Packet capture via scapy (optional pypcap)
- Redis Streams buffering for live ingestion
- Temporal Graph Network (TGN) model with memory + TransformerConv layers
- Dataset preparation for CIC-IDS2017 MachineLearningCSV exports
- FastAPI inference service with `/predict`, `/demo/predict`, `/health`, and WebSocket
- React dashboard wired to the inference API topology response
- Docker Compose stack for Redis, API, and dashboard
- GitHub Actions CI (pytest + black/flake8)
- Demo checkpoint helper: `scripts/create_demo_model.py`

### Fixed
- TGN time encoding now uses tensors (forward pass works)
- Dataset mapper `preserve_all_features` NameError
- CI install failure caused by hard dependency on `pypcap`
- Inference startup no longer hard-requires Redis or a trained checkpoint

### Notes
- v0.1.0 is a research / beta release. Train a model before production use.
- Packet capture still requires elevated privileges and a live interface.

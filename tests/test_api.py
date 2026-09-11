"""API smoke tests using FastAPI TestClient."""

from fastapi.testclient import TestClient

import src.inference.api as api_module
from src.inference.api import InferenceService, app


def test_health_and_predict(tmp_path):
    config_path = tmp_path / "inference_config.yaml"
    config_path.write_text(
        """
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
redis:
  enabled: false
  host: "localhost"
  port: 6379
  stream_name: "network_flows"
  consumer_group: "inference_workers"
  batch_size: 10
  block_time: 100
model:
  path: "./models/missing.pt"
  device: "cpu"
  threshold: 0.5
explainability:
  enable_shap: false
"""
    )
    service = InferenceService(config_path=str(config_path))

    with TestClient(app) as client:
        api_module.inference_service = service

        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["model_loaded"] is True
        assert body["redis_connected"] is False

        payload = [
            {
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 1234,
                "dst_port": 80,
                "protocol": 6,
                "timestamp": 1.0,
                "packet_size": 100,
            },
            {
                "src_ip": "10.0.0.2",
                "dst_ip": "10.0.0.3",
                "src_port": 80,
                "dst_port": 443,
                "protocol": 6,
                "timestamp": 2.0,
                "packet_size": 200,
            },
        ]
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "anomaly_score" in data
        assert "is_anomaly" in data
        assert data["topology"]["nodes"]
        assert data["topology"]["links"]

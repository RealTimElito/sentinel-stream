"""FastAPI inference service."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..ml.explainability import TGNExplainer
from ..ml.graph_construction import (
    NetworkGraphBuilder,
    aggregate_flows_by_time_window,
)
from ..ml.tgn_model import TemporalGraphNetwork
from ..utils.feature_engineering import extract_flow_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

inference_service: Optional["InferenceService"] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global inference_service
    try:
        inference_service = InferenceService()
        if inference_service.redis_buffer is not None:
            asyncio.create_task(inference_service.process_redis_stream())
        logger.info("Inference service started")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start inference service: %s", exc)
    yield
    inference_service = None


app = FastAPI(
    title="Sentinel-Stream Inference API",
    version="0.1.0",
    description="Real-time TGN anomaly scoring for network flows",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowData(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6
    timestamp: float
    packet_size: int = Field(default=64, ge=0)
    tcp_flags: Optional[int] = None


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    explanation: Optional[Dict[str, Any]] = None
    topology: Optional[Dict[str, Any]] = None


class InferenceService:
    """Inference service for real-time predictions."""

    def __init__(self, config_path: str = "configs/inference_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.device = torch.device(self.config["model"].get("device", "cpu"))
        self.model_trained = False
        self.model = self._load_model()
        self.model.eval()
        self.graph_builder = NetworkGraphBuilder(
            time_window=self.config.get("data", {}).get("time_window", 60.0)
        )
        self.threshold = float(self.config["model"]["threshold"])
        self.redis_buffer = None
        self._init_redis()

    def _init_redis(self) -> None:
        redis_config = self.config.get("redis", {})
        if not redis_config.get("enabled", True):
            logger.info("Redis stream consumer disabled by config")
            return
        try:
            from ..ingestion.redis_buffer import RedisBuffer

            self.redis_buffer = RedisBuffer(
                host=os.getenv("REDIS_HOST", redis_config.get("host", "localhost")),
                port=int(os.getenv("REDIS_PORT", redis_config.get("port", 6379))),
                stream_name=redis_config.get("stream_name", "network_flows"),
            )
        except Exception as exc:  # noqa: BLE001 - startup should stay up without Redis
            logger.warning("Redis unavailable (%s); stream consumer disabled", exc)
            self.redis_buffer = None

    def _load_model(self) -> TemporalGraphNetwork:
        model_path = os.getenv(
            "MODEL_PATH", self.config["model"].get("path", "./models/tgn_model.pt")
        )
        model = TemporalGraphNetwork(
            node_dim=64,
            edge_dim=32,
            time_dim=16,
            memory_dim=64,
            message_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state)
            self.model_trained = True
            logger.info("Model loaded from %s", model_path)
        else:
            logger.warning(
                "Model file not found at %s; using randomly initialized weights. "
                "Run: python scripts/create_demo_model.py",
                model_path,
            )
        return model.to(self.device)

    def _flows_to_dicts(self, flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for flow in flows:
            item = dict(flow)
            if "size" not in item:
                item["size"] = item.get("packet_size", 0)
            if "packet_size" not in item:
                item["packet_size"] = item.get("size", 0)
            normalized.append(item)
        return normalized

    def _build_topology(
        self, flows: List[Dict[str, Any]], is_anomaly: bool, score: float
    ) -> Dict[str, Any]:
        nodes = {}
        links = []
        for flow in flows:
            src = str(flow["src_ip"])
            dst = str(flow["dst_ip"])
            nodes[src] = {
                "id": src,
                "label": src,
                "group": 0,
                "anomaly": bool(is_anomaly),
            }
            nodes[dst] = {
                "id": dst,
                "label": dst,
                "group": 1,
                "anomaly": bool(is_anomaly),
            }
            links.append(
                {
                    "source": src,
                    "target": dst,
                    "value": max(score, 0.05),
                    "anomaly": bool(is_anomaly),
                }
            )
        return {"nodes": list(nodes.values()), "links": links}

    def predict(self, flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not flows:
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "explanation": None,
                "topology": {"nodes": [], "links": []},
            }

        flows = self._flows_to_dicts(flows)
        aggregated_flows = aggregate_flows_by_time_window(flows)
        for flow in aggregated_flows:
            flow["features"] = extract_flow_features(flow.get("packets", []))

        graph = self.graph_builder.build_temporal_graph(aggregated_flows).to(self.device)
        with torch.no_grad():
            predictions, _ = self.model(graph)
            avg_score = float(torch.sigmoid(predictions.mean()).item())

        is_anomaly = avg_score > self.threshold
        result: Dict[str, Any] = {
            "anomaly_score": avg_score,
            "is_anomaly": is_anomaly,
            "explanation": None,
            "topology": self._build_topology(aggregated_flows, is_anomaly, avg_score),
        }

        if self.config.get("explainability", {}).get("enable_shap", False):
            try:
                explainer = TGNExplainer(self.model, device=str(self.device))
                explanation = explainer.explain_graph(graph.cpu())
                result["explanation"] = {
                    "top_nodes": explanation["node_importance"].tolist(),
                    "top_edges": explanation["edge_importance"].tolist(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate explanation: %s", exc)

        return result

    async def process_redis_stream(self) -> None:
        if self.redis_buffer is None:
            return

        consumer_group = self.config["redis"]["consumer_group"]
        consumer_name = "inference_worker_1"
        batch_size = self.config["redis"]["batch_size"]
        block_time = self.config["redis"]["block_time"]
        flows_buffer: List[Dict[str, Any]] = []

        while True:
            try:
                messages = self.redis_buffer.read_messages(
                    consumer_group=consumer_group,
                    consumer_name=consumer_name,
                    count=batch_size,
                    block=block_time,
                )
                for msg in messages:
                    fields = msg["fields"]
                    flows_buffer.append(
                        {
                            "src_ip": fields["src_ip"],
                            "dst_ip": fields["dst_ip"],
                            "src_port": int(fields["src_port"]),
                            "dst_port": int(fields["dst_port"]),
                            "protocol": int(fields["protocol"]),
                            "timestamp": float(fields["timestamp"]),
                            "size": int(fields["packet_size"]),
                            "packet_size": int(fields["packet_size"]),
                        }
                    )
                    if len(flows_buffer) >= batch_size:
                        result = self.predict(flows_buffer)
                        logger.info(
                            "Processed %s flows, anomaly=%s score=%.4f",
                            len(flows_buffer),
                            result["is_anomaly"],
                            result["anomaly_score"],
                        )
                        flows_buffer = []
                    self.redis_buffer.acknowledge(consumer_group, msg["id"])
                await asyncio.sleep(0.1)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing Redis stream: %s", exc)
                await asyncio.sleep(1)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Sentinel-Stream Inference API", "status": "running"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy" if inference_service is not None else "degraded",
        "model_loaded": inference_service is not None,
        "model_trained": bool(inference_service.model_trained if inference_service else False),
        "redis_connected": bool(
            inference_service.redis_buffer is not None if inference_service else False
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(flows: List[FlowData]) -> PredictionResponse:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    flows_dict = [flow.model_dump() for flow in flows]
    try:
        result = inference_service.predict(flows_dict)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/demo/predict")
async def demo_predict() -> PredictionResponse:
    """Score a small synthetic flow batch (useful for dashboard smoke tests)."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    import time

    now = time.time()
    flows = [
        {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 44321,
            "dst_port": 80,
            "protocol": 6,
            "timestamp": now,
            "packet_size": 120,
        },
        {
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.3",
            "src_port": 80,
            "dst_port": 443,
            "protocol": 6,
            "timestamp": now + 0.2,
            "packet_size": 900,
        },
        {
            "src_ip": "10.0.0.8",
            "dst_ip": "10.0.0.1",
            "src_port": 22,
            "dst_port": 51515,
            "protocol": 6,
            "timestamp": now + 0.4,
            "packet_size": 64,
        },
    ]
    return PredictionResponse(**inference_service.predict(flows))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if inference_service is None:
                await websocket.send_json({"error": "Inference service not initialized"})
                continue
            if "flows" in data:
                result = inference_service.predict(data["flows"])
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Invalid request format"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.error("WebSocket error: %s", exc)
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    with open("configs/inference_config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    server_config = config["server"]
    uvicorn.run(
        app,
        host=server_config["host"],
        port=server_config["port"],
        log_level=server_config["log_level"],
    )

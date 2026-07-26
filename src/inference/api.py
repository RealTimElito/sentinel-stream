"""FastAPI inference service."""

import asyncio
import logging
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import yaml
import json

from ..ml.tgn_model import TemporalGraphNetwork
from ..ml.graph_construction import NetworkGraphBuilder, aggregate_flows_by_time_window
from ..ingestion.redis_buffer import RedisBuffer
from ..utils.feature_engineering import extract_flow_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel-Stream Inference API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowData(BaseModel):
    """Flow data model."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    timestamp: float
    packet_size: int
    tcp_flags: Optional[int] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    anomaly_score: float
    is_anomaly: bool
    explanation: Optional[Dict] = None


class InferenceService:
    """Inference service for real-time predictions."""
    
    def __init__(self, config_path: str = "configs/inference_config.yaml"):
        """Initialize inference service."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.device = torch.device(self.config['model']['device'])
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize graph builder
        self.graph_builder = NetworkGraphBuilder(
            time_window=self.config.get('data', {}).get('time_window', 60.0)
        )
        
        # Initialize Redis buffer
        redis_config = self.config['redis']
        self.redis_buffer = RedisBuffer(
            host=redis_config['host'],
            port=redis_config['port'],
            stream_name=redis_config['stream_name']
        )
        
        self.threshold = self.config['model']['threshold']
    
    def _load_model(self) -> TemporalGraphNetwork:
        """Load trained model."""
        model_path = self.config['model']['path']
        
        # Create model with default config
        model = TemporalGraphNetwork(
            node_dim=64,
            edge_dim=32,
            time_dim=16,
            memory_dim=64,
            message_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file not found at {model_path}, using untrained model")
        
        return model.to(self.device)
    
    def predict(self, flows: List[Dict]) -> Dict:
        """
        Predict anomalies for flows.
        
        Args:
            flows: List of flow dictionaries
            
        Returns:
            Prediction results
        """
        if not flows:
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'explanation': None
            }
        
        try:
            # Aggregate flows
            aggregated_flows = aggregate_flows_by_time_window(flows)
            
            # Extract features
            for flow in aggregated_flows:
                flow['features'] = extract_flow_features(flow.get('packets', []))
            
            # Build graph
            graph = self.graph_builder.build_temporal_graph(aggregated_flows)
            graph = graph.to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions, _ = self.model(graph)
                avg_score = torch.sigmoid(predictions.mean()).item()
            
            is_anomaly = avg_score > self.threshold
            
            result = {
                'anomaly_score': avg_score,
                'is_anomaly': is_anomaly,
                'explanation': None
            }
            
            # Add explainability if enabled
            if self.config.get('explainability', {}).get('enable_shap', False):
                try:
                    from ..ml.explainability import TGNExplainer
                    explainer = TGNExplainer(self.model, device=str(self.device))
                    explanation = explainer.explain_graph(graph.cpu())
                    result['explanation'] = {
                        'top_nodes': explanation['node_importance'].tolist(),
                        'top_edges': explanation['edge_importance'].tolist()
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate explanation: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_redis_stream(self):
        """Process messages from Redis stream."""
        consumer_group = self.config['redis']['consumer_group']
        consumer_name = "inference_worker_1"
        batch_size = self.config['redis']['batch_size']
        block_time = self.config['redis']['block_time']
        
        flows_buffer = []
        
        while True:
            try:
                messages = self.redis_buffer.read_messages(
                    consumer_group=consumer_group,
                    consumer_name=consumer_name,
                    count=batch_size,
                    block=block_time
                )
                
                for msg in messages:
                    fields = msg['fields']
                    
                    flow = {
                        'src_ip': fields['src_ip'],
                        'dst_ip': fields['dst_ip'],
                        'src_port': int(fields['src_port']),
                        'dst_port': int(fields['dst_port']),
                        'protocol': int(fields['protocol']),
                        'timestamp': float(fields['timestamp']),
                        'size': int(fields['packet_size']),
                    }
                    
                    flows_buffer.append(flow)
                    
                    # Process when buffer reaches threshold
                    if len(flows_buffer) >= batch_size:
                        result = self.predict(flows_buffer)
                        logger.info(f"Processed {len(flows_buffer)} flows, anomaly: {result['is_anomaly']}")
                        flows_buffer = []
                    
                    # Acknowledge message
                    self.redis_buffer.acknowledge(consumer_group, msg['id'])
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing Redis stream: {e}")
                await asyncio.sleep(1)


# Global inference service instance
inference_service: Optional[InferenceService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference service on startup."""
    global inference_service
    try:
        inference_service = InferenceService()
        # Start Redis stream processing in background
        asyncio.create_task(inference_service.process_redis_stream())
        logger.info("Inference service started")
    except Exception as e:
        logger.error(f"Failed to start inference service: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sentinel-Stream Inference API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_service is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(flows: List[FlowData]):
    """
    Predict anomalies for flows.
    
    Args:
        flows: List of flow data
        
    Returns:
        Prediction results
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    
    flows_dict = [flow.dict() for flow in flows]
    result = inference_service.predict(flows_dict)
    
    return PredictionResponse(**result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'flows' in data:
                flows = data['flows']
                result = inference_service.predict(flows)
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Invalid request format"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    config = yaml.safe_load(open("configs/inference_config.yaml"))
    server_config = config['server']
    
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        log_level=server_config['log_level']
    )


# Sentinel-Stream Architecture

## System Overview

Sentinel-Stream is a real-time network anomaly detection system that uses Temporal Graph Networks (TGN) to identify security threats in network traffic.

## Components

### 1. Data Ingestion Layer

**Packet Capture (`src/ingestion/capture.py`)**
- Uses libpcap (or scapy fallback) to capture network packets
- Extracts metadata: IP addresses, ports, protocols, packet sizes
- Supports BPF filtering for selective capture

**Redis Buffer (`src/ingestion/redis_buffer.py`)**
- Streams packet metadata to Redis Streams
- Provides reliable message queuing
- Supports consumer groups for parallel processing

**Secure Transmission (`scripts/secure_transmission.sh`)**
- Uses socat with OpenSSL for encrypted data transfer
- Enables secure communication between capture and inference nodes
- Self-signed certificate generation included

### 2. Machine Learning Core

**Graph Construction (`src/ml/graph_construction.py`)**
- Converts network flows into temporal graphs
- IP addresses → Nodes
- Network flows → Directed, timestamped Edges
- Aggregates flows within time windows

**TGN Model (`src/ml/tgn_model.py`)**
- Temporal Graph Network implementation
- Memory module for node state tracking
- Message passing between nodes
- Self-supervised pre-training (next edge prediction)
- Supervised fine-tuning (anomaly classification)

**Training Pipeline (`src/ml/training.py`)**
- Two-stage training:
  1. Self-supervised pre-training
  2. Supervised fine-tuning
- Early stopping and validation
- Model checkpointing

**Explainability (`src/ml/explainability.py`)**
- SHAP integration for model interpretability
- Identifies contributing nodes and edges
- Provides attack attribution

### 3. Inference Engine

**FastAPI Service (`src/inference/api.py`)**
- RESTful API for predictions
- WebSocket support for real-time updates
- Processes Redis Streams in background
- Integrates SHAP explanations

### 4. Dashboard

**React Frontend (`dashboard/`)**
- Real-time network topology visualization
- Force-directed graph layout
- Statistics panel
- Alert panel for detected anomalies
- WebSocket connection for live updates

## Data Flow

```
Network Traffic
    ↓
Packet Capture (libpcap)
    ↓
Feature Extraction
    ↓
Redis Streams
    ↓
Graph Construction
    ↓
TGN Model Inference
    ↓
Anomaly Detection
    ↓
Dashboard Visualization
```

## Security Considerations

1. **Secure Transmission**: OpenSSL encryption for data in transit
2. **Access Control**: Redis authentication (recommended for production)
3. **Network Isolation**: Capture node can be isolated from inference node
4. **Certificate Management**: Self-signed certs for development, CA-signed for production

## Scalability

- **Horizontal Scaling**: Multiple inference workers can consume from Redis Streams
- **Vertical Scaling**: GPU support for model inference
- **Load Balancing**: Multiple FastAPI instances behind a load balancer

## Deployment

See `docker-compose.yml` for containerized deployment or `docs/DEPLOYMENT.md` for detailed instructions.


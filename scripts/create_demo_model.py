#!/usr/bin/env python3
"""Create a small demo TGN checkpoint for local inference smoke tests."""

from __future__ import annotations

import argparse
import logging
import os
import random

import torch
from torch.utils.data import DataLoader

from src.ml.graph_construction import NetworkGraphBuilder
from src.ml.tgn_model import TemporalGraphNetwork
from src.ml.training import TemporalGraphDataset, TGNTrainer, collate_temporal_batch
from src.utils.feature_engineering import extract_flow_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _synthetic_flows(n_flows: int = 40, seed: int = 0):
    rng = random.Random(seed)
    flows = []
    labels = []
    for i in range(n_flows):
        anomalous = rng.random() < 0.25
        src = f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 50)}"
        dst = f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 50)}"
        size = rng.randint(800, 1500) if anomalous else rng.randint(40, 200)
        flows.append(
            {
                "src_ip": src,
                "dst_ip": dst,
                "src_port": rng.randint(1024, 65535),
                "dst_port": 80 if not anomalous else rng.randint(1, 1023),
                "protocol": 6,
                "timestamp": float(i) + rng.random(),
                "packet_size": size,
                "size": size,
                "packet_count": rng.randint(1, 20),
                "total_bytes": size * rng.randint(1, 10),
                "duration": rng.random(),
                "packets": [
                    {"timestamp": float(i), "size": size, "protocol": 6},
                    {"timestamp": float(i) + 0.1, "size": size, "protocol": 6},
                ],
            }
        )
        labels.append(1 if anomalous else 0)
    return flows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo TGN checkpoint")
    parser.add_argument("--output", default="./models/tgn_model.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--graphs", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    builder = NetworkGraphBuilder()
    graphs = []
    graph_labels = []
    for i in range(args.graphs):
        flows, labels = _synthetic_flows(n_flows=30, seed=i)
        for flow in flows:
            flow["features"] = extract_flow_features(flow.get("packets", []))
        graphs.append(builder.build_temporal_graph(flows))
        graph_labels.append(int(sum(labels) > len(labels) // 4))

    dataset = TemporalGraphDataset(graphs, graph_labels)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_temporal_batch,
    )

    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = "cuda"
        except RuntimeError:
            logger.warning("CUDA available but unusable; falling back to CPU")
    model = TemporalGraphNetwork()
    trainer = TGNTrainer(model, device=device, learning_rate=1e-3)
    trainer.pretrain(loader, num_epochs=max(1, args.epochs // 2), verbose=False)
    trainer.train(loader, num_epochs=args.epochs, verbose=False)
    trainer.save_model(args.output)
    logger.info("Wrote demo checkpoint to %s", args.output)


if __name__ == "__main__":
    main()

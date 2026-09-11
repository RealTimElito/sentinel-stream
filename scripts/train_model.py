#!/usr/bin/env python3
"""Train TGN model on processed flow CSV data."""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ml.graph_construction import NetworkGraphBuilder, aggregate_flows_by_time_window
from src.ml.tgn_model import TemporalGraphNetwork
from src.ml.training import (
    TemporalGraphDataset,
    TGNTrainer,
    collate_temporal_batch,
    load_config,
)
from src.utils.feature_engineering import extract_flow_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str):
    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)
    flows = []
    for _, row in df.iterrows():
        flows.append(
            {
                "src_ip": row["src_ip"],
                "dst_ip": row["dst_ip"],
                "src_port": int(row["src_port"]),
                "dst_port": int(row["dst_port"]),
                "protocol": int(row["protocol"]),
                "timestamp": float(row["timestamp"]),
                "packet_size": int(row["packet_size"]),
                "size": int(row["packet_size"]),
            }
        )
    return flows, df["label"].astype(int).values


def create_graph(flows, labels, graph_builder):
    aggregated = aggregate_flows_by_time_window(flows)
    for flow in aggregated:
        flow["features"] = extract_flow_features(flow.get("packets", []))
    graph = graph_builder.build_temporal_graph(aggregated)
    label = int(sum(labels) > 0)
    return graph, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TGN model")
    parser.add_argument("--config", default="configs/tgn_config.yaml")
    parser.add_argument("--data-path", default="./data/processed/cic-ids2017_processed.csv")
    parser.add_argument("--output", default="./models")
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=64,
        help="Limit number of training graphs for faster iteration",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            logger.warning("CUDA available but unusable; falling back to CPU")
    logger.info("Using device: %s", device)

    flows, labels = load_training_data(args.data_path)
    graph_builder = NetworkGraphBuilder(
        time_window=data_config["time_window"],
        min_flow_duration=data_config["min_flow_duration"],
        max_flow_duration=data_config["max_flow_duration"],
    )

    graphs = []
    graph_labels = []
    batch_size = 100
    for i in range(0, len(flows), batch_size):
        if len(graphs) >= args.max_graphs:
            break
        batch_flows = flows[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        try:
            graph, label = create_graph(batch_flows, batch_labels, graph_builder)
            graphs.append(graph)
            graph_labels.append(label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create graph for batch %s: %s", i, exc)

    if not graphs:
        raise RuntimeError("No training graphs were created from the dataset")

    logger.info("Created %s graphs", len(graphs))
    dataset = TemporalGraphDataset(graphs, graph_labels)
    train_size = max(1, int(len(dataset) * (1 - training_config["validation_split"])))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_dataset = dataset
        val_loader = None
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(8, len(val_dataset)),
            shuffle=False,
            collate_fn=collate_temporal_batch,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(8, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_temporal_batch,
    )

    model = TemporalGraphNetwork(
        node_dim=model_config["node_dim"],
        edge_dim=model_config["edge_dim"],
        time_dim=model_config["time_dim"],
        memory_dim=model_config["memory_dim"],
        message_dim=model_config["message_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
    )
    trainer = TGNTrainer(
        model=model,
        device=str(device),
        learning_rate=training_config["learning_rate"],
    )

    logger.info("Starting pre-training...")
    trainer.pretrain(
        train_loader,
        num_epochs=min(5, training_config["pretrain_epochs"]),
    )
    logger.info("Starting fine-tuning...")
    trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=min(10, training_config["finetune_epochs"]),
        early_stopping_patience=training_config["early_stopping_patience"],
    )

    model_path = os.path.join(args.output, "tgn_model.pt")
    trainer.save_model(model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()

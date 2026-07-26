#!/usr/bin/env python3
"""Train TGN model."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import yaml
import logging

from src.ml.tgn_model import TemporalGraphNetwork
from src.ml.training import TGNTrainer, TemporalGraphDataset, load_config
from src.ml.graph_construction import NetworkGraphBuilder, aggregate_flows_by_time_window
from src.utils.feature_engineering import extract_flow_features
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str):
    """Load training data."""
    logger.info(f"Loading training data from {data_path}")
    
    # Load processed CSV
    df = pd.read_csv(data_path)
    
    # Convert to flow format
    flows = []
    for _, row in df.iterrows():
        flow = {
            'src_ip': row['src_ip'],
            'dst_ip': row['dst_ip'],
            'src_port': int(row['src_port']),
            'dst_port': int(row['dst_port']),
            'protocol': int(row['protocol']),
            'timestamp': float(row['timestamp']),
            'packet_size': int(row['packet_size']),
            'size': int(row['packet_size']),
        }
        flows.append(flow)
    
    return flows, df['label'].values


def create_graphs(flows, labels, graph_builder):
    """Create temporal graphs from flows."""
    logger.info("Creating temporal graphs...")
    
    # Aggregate flows
    aggregated_flows = aggregate_flows_by_time_window(flows)
    
    # Extract features
    for flow in aggregated_flows:
        flow['features'] = extract_flow_features(flow.get('packets', []))
    
    # Build graph
    graph = graph_builder.build_temporal_graph(aggregated_flows)
    
    return graph, labels[0] if len(labels) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Train TGN model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tgn_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/processed/cic-ids2017_processed.csv',
        help='Path to processed data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./models',
        help='Output directory for model'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    flows, labels = load_training_data(args.data_path)
    
    # Initialize graph builder
    graph_builder = NetworkGraphBuilder(
        time_window=data_config['time_window'],
        min_flow_duration=data_config['min_flow_duration'],
        max_flow_duration=data_config['max_flow_duration']
    )
    
    # Create graphs (simplified - in practice, create multiple graphs)
    logger.info("Creating graphs from flows...")
    graphs = []
    graph_labels = []
    
    # Split flows into batches for graph creation
    batch_size = 100
    for i in range(0, len(flows), batch_size):
        batch_flows = flows[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        try:
            graph, label = create_graphs(batch_flows, batch_labels, graph_builder)
            graphs.append(graph)
            graph_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Failed to create graph for batch {i}: {e}")
            continue
    
    logger.info(f"Created {len(graphs)} graphs")
    
    # Create dataset
    dataset = TemporalGraphDataset(graphs, graph_labels)
    train_size = int(len(dataset) * (1 - training_config['validation_split']))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False
    )
    
    # Create model
    model = TemporalGraphNetwork(
        node_dim=model_config['node_dim'],
        edge_dim=model_config['edge_dim'],
        time_dim=model_config['time_dim'],
        memory_dim=model_config['memory_dim'],
        message_dim=model_config['message_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout']
    )
    
    # Create trainer
    trainer = TGNTrainer(
        model=model,
        device=device,
        learning_rate=training_config['learning_rate']
    )
    
    # Pre-training (self-supervised)
    logger.info("Starting pre-training...")
    pretrain_losses = trainer.pretrain(
        train_loader,
        num_epochs=training_config['pretrain_epochs']
    )
    
    # Fine-tuning (supervised)
    logger.info("Starting fine-tuning...")
    train_losses = trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=training_config['finetune_epochs'],
        early_stopping_patience=training_config['early_stopping_patience']
    )
    
    # Save model
    model_path = os.path.join(args.output, 'tgn_model.pt')
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()


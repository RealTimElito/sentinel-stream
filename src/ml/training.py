"""Training pipeline for TGN model."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .tgn_model import TemporalGraphNetwork

logger = logging.getLogger(__name__)


def collate_temporal_batch(
    batch: List[Tuple],
) -> Tuple[List, torch.Tensor]:
    """Collate TemporalData samples without stacking heterogeneous graphs."""
    graphs = []
    labels = []
    for item in batch:
        if isinstance(item, tuple):
            graph, label = item
            graphs.append(graph)
            labels.append(label)
        else:
            graphs.append(item)
            labels.append(0)
    return graphs, torch.tensor(labels, dtype=torch.float32)


class TemporalGraphDataset(Dataset):
    """Dataset of temporal graphs with optional labels."""

    def __init__(self, graphs: List, labels: Optional[List] = None):
        self.graphs = graphs
        self.labels = labels if labels is not None else [0] * len(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx]


class TGNTrainer:
    """Trainer for Temporal Graph Network."""

    def __init__(
        self,
        model: TemporalGraphNetwork,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.pretrain_criterion = nn.MSELoss()

    def pretrain(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        verbose: bool = True,
    ) -> List[float]:
        self.model.train()
        losses: List[float] = []

        for epoch in range(num_epochs):
            epoch_losses = []
            iterator = (
                tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
                if verbose
                else train_loader
            )

            for graphs, _ in iterator:
                batch_loss = 0.0
                self.optimizer.zero_grad()
                for graph in graphs:
                    graph = graph.to(self.device)
                    predictions = self.model.predict_next_edge(graph)
                    targets = torch.ones_like(predictions)
                    loss = self.pretrain_criterion(predictions, targets)
                    loss.backward()
                    batch_loss += float(loss.item())
                self.optimizer.step()
                epoch_losses.append(batch_loss / max(len(graphs), 1))
                if verbose:
                    iterator.set_postfix({"loss": epoch_losses[-1]})

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(avg_loss)
            if verbose:
                logger.info(
                    "Pretrain Epoch %s/%s, Loss: %.4f",
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                )
        return losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        self.model.train()
        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_train_losses = []
            iterator = (
                tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
                if verbose
                else train_loader
            )

            for graphs, labels in iterator:
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                batch_loss = 0.0
                for graph, label in zip(graphs, labels):
                    graph = graph.to(self.device)
                    predictions, _ = self.model(graph)
                    graph_logit = predictions.mean()
                    loss = self.criterion(graph_logit, label)
                    loss.backward()
                    batch_loss += float(loss.item())
                self.optimizer.step()
                epoch_train_losses.append(batch_loss / max(len(graphs), 1))
                if verbose:
                    iterator.set_postfix({"loss": epoch_train_losses[-1]})

            avg_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0
            train_losses.append(avg_train_loss)

            if val_loader:
                val_loss = self.validate(val_loader, verbose=False)
                val_losses.append(val_loss)
                if verbose:
                    logger.info(
                        "Epoch %s/%s, Train Loss: %.4f, Val Loss: %.4f",
                        epoch + 1,
                        num_epochs,
                        avg_train_loss,
                        val_loss,
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping at epoch %s", epoch + 1)
                        break
            elif verbose:
                logger.info(
                    "Epoch %s/%s, Train Loss: %.4f",
                    epoch + 1,
                    num_epochs,
                    avg_train_loss,
                )

        return {"train_losses": train_losses, "val_losses": val_losses}

    def validate(self, val_loader: DataLoader, verbose: bool = True) -> float:
        self.model.eval()
        losses = []
        iterator = tqdm(val_loader, desc="Validation") if verbose else val_loader

        with torch.no_grad():
            for graphs, labels in iterator:
                labels = labels.to(self.device)
                for graph, label in zip(graphs, labels):
                    graph = graph.to(self.device)
                    predictions, _ = self.model(graph)
                    graph_logit = predictions.mean()
                    loss = self.criterion(graph_logit, label)
                    losses.append(float(loss.item()))

        self.model.train()
        return float(np.mean(losses)) if losses else 0.0

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": {
                    "node_dim": self.model.node_dim,
                    "edge_dim": self.model.edge_dim,
                    "time_dim": self.model.time_dim,
                    "memory_dim": self.model.memory_dim,
                    "message_dim": self.model.message_dim,
                },
            },
            path,
        )
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Model loaded from %s", path)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

"""Training pipeline for TGN model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm
import yaml

from .tgn_model import TemporalGraphNetwork
from .graph_construction import NetworkGraphBuilder, aggregate_flows_by_time_window

logger = logging.getLogger(__name__)


class TemporalGraphDataset(Dataset):
    """Dataset for temporal graph data."""
    
    def __init__(self, graphs: List, labels: Optional[List] = None):
        """
        Initialize dataset.
        
        Args:
            graphs: List of TemporalData objects
            labels: Optional list of labels (for supervised learning)
        """
        self.graphs = graphs
        self.labels = labels if labels is not None else [0] * len(graphs)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.labels:
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx]


class TGNTrainer:
    """Trainer for Temporal Graph Network."""
    
    def __init__(
        self,
        model: TemporalGraphNetwork,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: TGN model instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.pretrain_criterion = nn.MSELoss()
    
    def pretrain(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        verbose: bool = True
    ) -> List[float]:
        """
        Self-supervised pre-training (predict next edge).
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs
            verbose: Whether to show progress
            
        Returns:
            List of training losses
        """
        self.model.train()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            if verbose:
                pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                if isinstance(batch, tuple):
                    graphs, _ = batch
                else:
                    graphs = batch
                
                if isinstance(graphs, list):
                    graphs = graphs[0]
                
                graphs = graphs.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Predict next edge
                predictions = self.model.predict_next_edge(graphs)
                
                # Create target (simplified: predict if edge exists)
                # In practice, use negative sampling
                targets = torch.ones_like(predictions)
                
                loss = self.pretrain_criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if verbose:
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose:
                logger.info(f"Pretrain Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Supervised training for anomaly detection.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to show progress
            
        Returns:
            Dictionary with training and validation losses
        """
        self.model.train()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            epoch_train_losses = []
            
            if verbose:
                pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                graphs, labels = batch
                
                if isinstance(graphs, list):
                    graphs = graphs[0]
                
                graphs = graphs.to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, _ = self.model(graphs)
                
                # Average predictions for the graph
                if len(predictions.shape) > 1:
                    predictions = predictions.mean(dim=0)
                
                # Ensure predictions and labels have compatible shapes
                if predictions.shape != labels.shape:
                    if len(predictions.shape) == 0:
                        predictions = predictions.unsqueeze(0)
                    if len(labels.shape) == 0:
                        labels = labels.unsqueeze(0)
                    if predictions.shape[0] != labels.shape[0]:
                        predictions = predictions[:labels.shape[0]]
                
                loss = self.criterion(predictions.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_losses.append(loss.item())
                
                if verbose:
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self.validate(val_loader, verbose=False)
                val_losses.append(val_loss)
                
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def validate(self, val_loader: DataLoader, verbose: bool = True) -> float:
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for validation data
            verbose: Whether to show progress
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            if verbose:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
            
            for batch in pbar:
                graphs, labels = batch
                
                if isinstance(graphs, list):
                    graphs = graphs[0]
                
                graphs = graphs.to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                predictions, _ = self.model(graphs)
                
                if len(predictions.shape) > 1:
                    predictions = predictions.mean(dim=0)
                
                if predictions.shape != labels.shape:
                    if len(predictions.shape) == 0:
                        predictions = predictions.unsqueeze(0)
                    if len(labels.shape) == 0:
                        labels = labels.unsqueeze(0)
                    if predictions.shape[0] != labels.shape[0]:
                        predictions = predictions[:labels.shape[0]]
                
                loss = self.criterion(predictions.squeeze(), labels)
                losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses)
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


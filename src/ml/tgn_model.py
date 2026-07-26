"""Temporal Graph Network (TGN) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TemporalEncoding, TransformerConv
from torch_geometric.data import TemporalData
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MemoryModule(nn.Module):
    """Memory module for TGN."""
    
    def __init__(self, node_dim: int, memory_dim: int):
        """
        Initialize memory module.
        
        Args:
            node_dim: Dimension of node features
            memory_dim: Dimension of memory vectors
        """
        super().__init__()
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        
        # Memory vectors for each node
        self.register_buffer('memory', torch.zeros(1, memory_dim))
        
        # Memory update network
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
    
    def reset(self, num_nodes: int):
        """Reset memory for new graph."""
        self.memory = torch.zeros(num_nodes, self.memory_dim, device=self.memory.device)
    
    def update(self, node_ids: torch.Tensor, messages: torch.Tensor):
        """
        Update memory for given nodes.
        
        Args:
            node_ids: Node indices to update
            messages: Message vectors
        """
        if len(node_ids) == 0:
            return
        
        # Get current memory states
        current_memory = self.memory[node_ids]
        
        # Update memory
        updated_memory = self.memory_updater(messages, current_memory)
        
        # Store updated memory
        self.memory[node_ids] = updated_memory
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get memory states for nodes.
        
        Args:
            node_ids: Node indices
            
        Returns:
            Memory vectors
        """
        return self.memory[node_ids]


class MessageFunction(nn.Module):
    """Message computation function."""
    
    def __init__(self, memory_dim: int, edge_dim: int, message_dim: int):
        """
        Initialize message function.
        
        Args:
            memory_dim: Dimension of memory vectors
            edge_dim: Dimension of edge features
            message_dim: Dimension of output messages
        """
        super().__init__()
        self.message_dim = message_dim
        
        self.message_net = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
    
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages.
        
        Args:
            src_memory: Source node memory
            dst_memory: Destination node memory
            edge_attr: Edge attributes
            
        Returns:
            Message vectors
        """
        # Concatenate source memory, destination memory, and edge features
        message_input = torch.cat([src_memory, dst_memory, edge_attr], dim=-1)
        messages = self.message_net(message_input)
        return messages


class TemporalGraphNetwork(nn.Module):
    """Temporal Graph Network for anomaly detection."""
    
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        time_dim: int = 16,
        memory_dim: int = 64,
        message_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize TGN model.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            time_dim: Dimension of time encoding
            memory_dim: Dimension of memory vectors
            message_dim: Dimension of messages
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        
        # Time encoding
        self.time_encoder = TemporalEncoding(time_dim)
        
        # Memory module
        self.memory = MemoryModule(node_dim, memory_dim)
        
        # Message function
        self.message_function = MessageFunction(memory_dim, edge_dim, message_dim)
        
        # Embedding layers
        self.node_embedder = nn.Linear(node_dim, memory_dim)
        self.edge_embedder = nn.Linear(edge_dim + time_dim, edge_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                TransformerConv(
                    memory_dim,
                    memory_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim // 2, 1)
        )
        
        # Next edge prediction head (for pre-training)
        self.edge_predictor = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1)
        )
    
    def forward(
        self,
        data: TemporalData,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            data: TemporalData object
            return_embeddings: Whether to return node embeddings
            
        Returns:
            (anomaly_scores, node_embeddings)
        """
        # Reset memory
        num_nodes = data.x.size(0)
        self.memory.reset(num_nodes)
        
        # Process edges in temporal order
        src = data.src
        dst = data.dst
        t = data.t
        edge_attr = data.edge_attr
        x = data.x
        
        # Sort by timestamp
        sorted_indices = torch.argsort(t)
        src = src[sorted_indices]
        dst = dst[sorted_indices]
        t = t[sorted_indices]
        edge_attr = edge_attr[sorted_indices]
        
        # Embed nodes
        node_emb = self.node_embedder(x)
        
        # Process each edge
        anomaly_scores = []
        node_embeddings = []
        
        for i in range(len(src)):
            src_id = src[i].item()
            dst_id = dst[i].item()
            edge_feat = edge_attr[i]
            timestamp = t[i].item()
            
            # Get current memory states
            src_memory = self.memory.get_memory(torch.tensor([src_id], device=x.device))
            dst_memory = self.memory.get_memory(torch.tensor([dst_id], device=x.device))
            
            # Encode time
            time_enc = self.time_encoder(timestamp)
            
            # Combine edge features with time encoding
            edge_feat_with_time = torch.cat([edge_feat, time_enc], dim=-1)
            edge_feat_encoded = self.edge_embedder(edge_feat_with_time)
            
            # Compute message
            message = self.message_function(src_memory, dst_memory, edge_feat_encoded)
            
            # Update memory
            self.memory.update(torch.tensor([src_id], device=x.device), message)
            self.memory.update(torch.tensor([dst_id], device=x.device), message)
            
            # Get updated memory
            src_memory_updated = self.memory.get_memory(torch.tensor([src_id], device=x.device))
            dst_memory_updated = self.memory.get_memory(torch.tensor([dst_id], device=x.device))
            
            # Compute anomaly score
            edge_representation = torch.cat([
                src_memory_updated.squeeze(),
                dst_memory_updated.squeeze(),
                edge_feat_encoded
            ], dim=-1)
            
            anomaly_score = self.classifier(edge_representation)
            anomaly_scores.append(anomaly_score)
            
            if return_embeddings:
                node_embeddings.append({
                    'src': src_memory_updated.squeeze().cpu(),
                    'dst': dst_memory_updated.squeeze().cpu(),
                    'edge': edge_feat_encoded.cpu()
                })
        
        anomaly_scores = torch.stack(anomaly_scores)
        
        if return_embeddings:
            return anomaly_scores, node_embeddings
        else:
            return anomaly_scores, None
    
    def predict_next_edge(
        self,
        data: TemporalData
    ) -> torch.Tensor:
        """
        Predict next edge (for self-supervised pre-training).
        
        Args:
            data: TemporalData object
            
        Returns:
            Prediction scores
        """
        # Forward pass to update memory
        _, _ = self.forward(data)
        
        # Get final memory states
        num_nodes = data.x.size(0)
        all_memories = self.memory.get_memory(torch.arange(num_nodes, device=data.x.device))
        
        # Predict next edge probabilities
        # For simplicity, predict probability of connection between all pairs
        # In practice, you'd use negative sampling
        src_emb = all_memories[data.src]
        dst_emb = all_memories[data.dst]
        
        edge_representation = torch.cat([src_emb, dst_emb], dim=-1)
        predictions = self.edge_predictor(edge_representation)
        
        return predictions.squeeze()


"""Temporal Graph Network (TGN) model for anomaly detection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import TemporalData
from torch_geometric.nn import TemporalEncoding, TransformerConv

logger = logging.getLogger(__name__)


class MemoryModule(nn.Module):
    """Per-node memory updated via GRU."""

    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_proj = (
            nn.Identity() if message_dim == memory_dim else nn.Linear(message_dim, memory_dim)
        )
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        # Runtime state only — never persisted in checkpoints.
        self.memory = torch.zeros(1, memory_dim)

    def reset(self, num_nodes: int, device: torch.device) -> None:
        self.memory = torch.zeros(num_nodes, self.memory_dim, device=device)

    def update(self, node_ids: torch.Tensor, messages: torch.Tensor) -> None:
        if node_ids.numel() == 0:
            return
        projected = self.message_proj(messages)
        current = self.memory[node_ids]
        self.memory[node_ids] = self.memory_updater(projected, current)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]


class MessageFunction(nn.Module):
    """Compute messages from source/destination memory and edge features."""

    def __init__(self, memory_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        message_input = torch.cat([src_memory, dst_memory, edge_attr], dim=-1)
        return self.message_net(message_input)


class TemporalGraphNetwork(nn.Module):
    """TGN-style encoder with memory, temporal edge encoding, and GNN layers."""

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        time_dim: int = 16,
        memory_dim: int = 64,
        message_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if memory_dim % num_heads != 0:
            raise ValueError("memory_dim must be divisible by num_heads")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        self.time_encoder = TemporalEncoding(time_dim)
        self.memory = MemoryModule(memory_dim, message_dim)
        self.message_function = MessageFunction(memory_dim, edge_dim, message_dim)

        self.node_embedder = nn.Linear(node_dim, memory_dim)
        self.edge_embedder = nn.Linear(edge_dim + time_dim, edge_dim)

        self.gnn_layers = nn.ModuleList(
            [
                TransformerConv(
                    memory_dim,
                    memory_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(memory_dim) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim // 2, 1),
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def _encode_edges(self, data: TemporalData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = data.src
        dst = data.dst
        t = data.t
        edge_attr = data.edge_attr

        order = torch.argsort(t)
        src = src[order]
        dst = dst[order]
        t = t[order]
        edge_attr = edge_attr[order]

        time_enc = self.time_encoder(t)
        edge_with_time = torch.cat([edge_attr, time_enc], dim=-1)
        edge_encoded = self.edge_embedder(edge_with_time)
        return src, dst, edge_encoded

    def _run_memory(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_encoded: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> None:
        device = node_emb.device
        self.memory.reset(node_emb.size(0), device)
        self.memory.memory = node_emb.detach().clone()
        for i in range(src.size(0)):
            src_id = src[i : i + 1]
            dst_id = dst[i : i + 1]
            edge_feat = edge_encoded[i : i + 1]
            src_mem = self.memory.get_memory(src_id)
            dst_mem = self.memory.get_memory(dst_id)
            message = self.message_function(src_mem, dst_mem, edge_feat)
            self.memory.update(src_id, message)
            self.memory.update(dst_id, message)

    def _apply_gnn(
        self,
        node_emb: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_encoded: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = torch.stack([src, dst], dim=0)
        h = node_emb
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h = conv(h, edge_index, edge_attr=edge_encoded)
            h = norm(h)
            h = torch.relu(h)
        return h

    def forward(
        self,
        data: TemporalData,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, List[Dict]]]]:
        """
        Forward pass.

        Returns:
            (anomaly_logits [num_edges, 1], optional embeddings)
        """
        device = data.x.device
        num_nodes = data.x.size(0)
        src, dst, edge_encoded = self._encode_edges(data)
        node_emb = self.node_embedder(data.x)

        self._run_memory(src, dst, edge_encoded, node_emb)
        memory_states = self.memory.get_memory(torch.arange(num_nodes, device=device))
        refined = self._apply_gnn(memory_states, src, dst, edge_encoded)

        scores = []
        embeddings: List[Dict] = []
        for i in range(src.size(0)):
            src_id = int(src[i].item())
            dst_id = int(dst[i].item())
            edge_feat = edge_encoded[i]
            edge_representation = torch.cat([refined[src_id], refined[dst_id], edge_feat], dim=-1)
            scores.append(self.classifier(edge_representation))
            if return_embeddings:
                embeddings.append(
                    {
                        "src": refined[src_id].detach().cpu(),
                        "dst": refined[dst_id].detach().cpu(),
                        "edge": edge_feat.detach().cpu(),
                    }
                )

        stacked = torch.stack(scores, dim=0)
        if return_embeddings:
            return stacked, embeddings
        return stacked, None

    def predict_next_edge(self, data: TemporalData) -> torch.Tensor:
        """Self-supervised next-edge scores for observed edges."""
        device = data.x.device
        num_nodes = data.x.size(0)
        src, dst, edge_encoded = self._encode_edges(data)
        node_emb = self.node_embedder(data.x)

        self._run_memory(src, dst, edge_encoded, node_emb)
        memories = self.memory.get_memory(torch.arange(num_nodes, device=device))
        refined = self._apply_gnn(memories, src, dst, edge_encoded)
        edge_representation = torch.cat([refined[src], refined[dst]], dim=-1)
        return self.edge_predictor(edge_representation).squeeze(-1)

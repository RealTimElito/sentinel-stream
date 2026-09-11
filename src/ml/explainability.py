"""Model explainability helpers for TGN predictions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import TemporalData

from .tgn_model import TemporalGraphNetwork

logger = logging.getLogger(__name__)


class TGNExplainer:
    """Lightweight attribution for TGN edge scores."""

    def __init__(self, model: TemporalGraphNetwork, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def explain_edge(
        self,
        data: TemporalData,
        edge_idx: int,
        background_data: Optional[List[TemporalData]] = None,
        max_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Attribute an edge score via simple feature occlusion.

        SHAP KernelExplainer is intentionally avoided here: the model takes a
        structured TemporalData object, not a flat feature matrix.
        """
        del background_data, max_samples
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            base_scores, _ = self.model(data)
            base = float(base_scores[edge_idx].item())

            edge_attr = data.edge_attr.clone()
            contributions = np.zeros(edge_attr.size(1), dtype=np.float64)
            for feat_idx in range(edge_attr.size(1)):
                masked = edge_attr.clone()
                masked[edge_idx, feat_idx] = 0.0
                masked_data = TemporalData(
                    src=data.src,
                    dst=data.dst,
                    t=data.t,
                    edge_attr=masked,
                    x=data.x,
                )
                scores, _ = self.model(masked_data)
                contributions[feat_idx] = base - float(scores[edge_idx].item())

            src_id = int(data.src[edge_idx].item())
            dst_id = int(data.dst[edge_idx].item())
            src_contribution = data.x[src_id].detach().cpu().numpy()
            dst_contribution = data.x[dst_id].detach().cpu().numpy()

        return {
            "src_node_contribution": src_contribution,
            "dst_node_contribution": dst_contribution,
            "edge_feature_contribution": contributions,
            "total_contribution": float(np.sum(np.abs(contributions))),
        }

    def explain_graph(
        self,
        data: TemporalData,
        background_data: Optional[List[TemporalData]] = None,
        max_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Explain a graph by edge score magnitude and incident node mass."""
        del background_data, max_samples
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            predictions, _ = self.model(data)
            edge_importance = torch.abs(predictions).squeeze(-1).cpu().numpy()

            num_nodes = data.x.shape[0]
            node_importance = np.zeros(num_nodes, dtype=np.float64)
            for i, (src, dst) in enumerate(zip(data.src, data.dst)):
                node_importance[int(src.item())] += edge_importance[i]
                node_importance[int(dst.item())] += edge_importance[i]

            if node_importance.max() > 0:
                node_importance = node_importance / node_importance.max()

            return {
                "node_importance": node_importance,
                "edge_importance": edge_importance,
                "predictions": predictions.detach().cpu().numpy(),
            }

"""SHAP integration for model explainability."""

import numpy as np
import torch
import shap
from typing import List, Dict, Optional, Tuple
import logging

from .tgn_model import TemporalGraphNetwork
from .graph_construction import TemporalData

logger = logging.getLogger(__name__)


class TGNExplainer:
    """SHAP-based explainer for TGN model."""
    
    def __init__(self, model: TemporalGraphNetwork, device: str = "cpu"):
        """
        Initialize explainer.
        
        Args:
            model: Trained TGN model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def explain_edge(
        self,
        data: TemporalData,
        edge_idx: int,
        background_data: Optional[List[TemporalData]] = None,
        max_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Explain a specific edge prediction.
        
        Args:
            data: TemporalData object
            edge_idx: Index of edge to explain
            background_data: Optional background data for SHAP
            max_samples: Maximum number of samples for SHAP
            
        Returns:
            Dictionary with SHAP values for nodes and edge features
        """
        self.model.eval()
        
        # Get edge information
        src_id = data.src[edge_idx].item()
        dst_id = data.dst[edge_idx].item()
        edge_attr = data.edge_attr[edge_idx]
        
        # Create wrapper function for SHAP
        def model_wrapper(node_features, edge_features):
            """Wrapper function for SHAP."""
            with torch.no_grad():
                # Create modified data
                modified_data = TemporalData(
                    src=data.src,
                    dst=data.dst,
                    t=data.t,
                    edge_attr=edge_features if isinstance(edge_features, torch.Tensor) else torch.tensor(edge_features),
                    x=node_features if isinstance(node_features, torch.Tensor) else torch.tensor(node_features)
                )
                
                # Get prediction
                predictions, _ = self.model(modified_data.to(self.device))
                
                # Return prediction for the specific edge
                return predictions[edge_idx].cpu().numpy()
        
        # Prepare background data
        if background_data is None:
            # Use current data as background
            background_node_features = data.x.unsqueeze(0)
            background_edge_features = data.edge_attr.unsqueeze(0)
        else:
            # Aggregate background data
            bg_nodes = [bg.x for bg in background_data]
            bg_edges = [bg.edge_attr for bg in background_data]
            
            # Pad to same size
            max_nodes = max(n.shape[0] for n in bg_nodes)
            max_edges = max(e.shape[0] for e in bg_edges)
            
            padded_nodes = []
            padded_edges = []
            
            for n, e in zip(bg_nodes, bg_edges):
                node_pad = torch.zeros(max_nodes - n.shape[0], n.shape[1])
                edge_pad = torch.zeros(max_edges - e.shape[0], e.shape[1])
                
                padded_nodes.append(torch.cat([n, node_pad], dim=0))
                padded_edges.append(torch.cat([e, edge_pad], dim=0))
            
            background_node_features = torch.stack(padded_nodes)
            background_edge_features = torch.stack(padded_edges)
        
        # Limit background samples
        if background_node_features.shape[0] > max_samples:
            indices = np.random.choice(
                background_node_features.shape[0],
                max_samples,
                replace=False
            )
            background_node_features = background_node_features[indices]
            background_edge_features = background_edge_features[indices]
        
        # Prepare instance to explain
        instance_node_features = data.x.unsqueeze(0)
        instance_edge_features = data.edge_attr.unsqueeze(0)
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model_wrapper,
            (
                background_node_features.numpy(),
                background_edge_features.numpy()
            )
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(
            (
                instance_node_features.numpy(),
                instance_edge_features.numpy()
            ),
            nsamples=max_samples
        )
        
        # Extract node and edge contributions
        node_shap_values = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
        edge_shap_values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[1]
        
        # Get contributions for source and destination nodes
        src_contribution = node_shap_values[src_id] if src_id < len(node_shap_values) else np.zeros(node_shap_values.shape[1])
        dst_contribution = node_shap_values[dst_id] if dst_id < len(node_shap_values) else np.zeros(node_shap_values.shape[1])
        
        # Get edge feature contributions
        edge_contribution = edge_shap_values[edge_idx] if edge_idx < len(edge_shap_values) else np.zeros(edge_shap_values.shape[1])
        
        return {
            'src_node_contribution': src_contribution,
            'dst_node_contribution': dst_contribution,
            'edge_feature_contribution': edge_contribution,
            'total_contribution': np.sum(np.abs(src_contribution)) + np.sum(np.abs(dst_contribution)) + np.sum(np.abs(edge_contribution))
        }
    
    def explain_graph(
        self,
        data: TemporalData,
        background_data: Optional[List[TemporalData]] = None,
        max_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Explain entire graph prediction.
        
        Args:
            data: TemporalData object
            background_data: Optional background data
            max_samples: Maximum number of samples
            
        Returns:
            Dictionary with node and edge importance scores
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions
            predictions, embeddings = self.model(data.to(self.device), return_embeddings=True)
            
            # Compute importance based on prediction magnitude
            edge_importance = torch.abs(predictions).cpu().numpy()
            
            # Aggregate node importance from edges
            num_nodes = data.x.shape[0]
            node_importance = np.zeros(num_nodes)
            
            for i, (src, dst) in enumerate(zip(data.src, data.dst)):
                node_importance[src.item()] += edge_importance[i]
                node_importance[dst.item()] += edge_importance[i]
            
            # Normalize
            if node_importance.max() > 0:
                node_importance = node_importance / node_importance.max()
            
            return {
                'node_importance': node_importance,
                'edge_importance': edge_importance,
                'predictions': predictions.cpu().numpy()
            }


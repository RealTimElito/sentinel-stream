"""Graph construction from network flows."""

import numpy as np
import torch
from torch_geometric.data import Data, TemporalData
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class NetworkGraphBuilder:
    """Build temporal graphs from network flow data."""
    
    def __init__(
        self,
        time_window: float = 60.0,
        min_flow_duration: float = 0.1,
        max_flow_duration: float = 3600.0
    ):
        """
        Initialize graph builder.
        
        Args:
            time_window: Time window for graph construction (seconds)
            min_flow_duration: Minimum flow duration to include
            max_flow_duration: Maximum flow duration to include
        """
        self.time_window = time_window
        self.min_flow_duration = min_flow_duration
        self.max_flow_duration = max_flow_duration
    
    def ip_to_node_id(self, ip: str, node_map: Dict[str, int]) -> int:
        """
        Map IP address or flow identifier to node ID.
        
        Args:
            ip: IP address or flow identifier (e.g., 'port_80', 'flow_123_src')
            node_map: Dictionary mapping identifiers to node IDs
            
        Returns:
            Node ID
        """
        if ip not in node_map:
            node_map[ip] = len(node_map)
        return node_map[ip]
    
    def build_temporal_graph(
        self,
        flows: List[Dict],
        node_features: Optional[Dict[str, np.ndarray]] = None
    ) -> TemporalData:
        """
        Build temporal graph from network flows.
        
        Args:
            flows: List of flow dictionaries with:
                - src_ip, dst_ip
                - timestamp
                - features (packet sizes, inter-arrival times, etc.)
            node_features: Optional pre-computed node features
            
        Returns:
            TemporalData object
        """
        if not flows:
            raise ValueError("Empty flow list")
        
        # Sort flows by timestamp
        sorted_flows = sorted(flows, key=lambda x: x['timestamp'])
        
        # Build node mapping
        node_map = {}
        for flow in sorted_flows:
            self.ip_to_node_id(flow['src_ip'], node_map)
            self.ip_to_node_id(flow['dst_ip'], node_map)
        
        num_nodes = len(node_map)
        
        # Extract edge information
        src_nodes = []
        dst_nodes = []
        timestamps = []
        edge_features = []
        
        for flow in sorted_flows:
            src_id = self.ip_to_node_id(flow['src_ip'], node_map)
            dst_id = self.ip_to_node_id(flow['dst_ip'], node_map)
            
            src_nodes.append(src_id)
            dst_nodes.append(dst_id)
            timestamps.append(flow['timestamp'])
            
            # Extract edge features
            edge_feat = self._extract_edge_features(flow)
            edge_features.append(edge_feat)
        
        # Convert to tensors
        src = torch.tensor(src_nodes, dtype=torch.long)
        dst = torch.tensor(dst_nodes, dtype=torch.long)
        t = torch.tensor(timestamps, dtype=torch.float)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create node features if not provided
        if node_features is None:
            x = self._create_node_features(node_map, sorted_flows)
        else:
            x = torch.tensor([
                node_features.get(ip, np.zeros(64))
                for ip in sorted(node_map.keys())
            ], dtype=torch.float)
        
        # Create temporal data
        temporal_data = TemporalData(
            src=src,
            dst=dst,
            t=t,
            edge_attr=edge_attr,
            x=x
        )
        
        return temporal_data
    
    def _extract_edge_features(self, flow: Dict) -> np.ndarray:
        """
        Extract features for an edge.
        
        Args:
            flow: Flow dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic flow features
        features.append(flow.get('packet_count', 0))
        features.append(flow.get('total_bytes', 0))
        features.append(flow.get('duration', 0.0))
        
        # Protocol
        features.append(flow.get('protocol', 0))
        
        # Port information
        features.append(flow.get('src_port', 0))
        features.append(flow.get('dst_port', 0))
        
        # Statistical features
        if 'features' in flow:
            feat_dict = flow['features']
            features.extend([
                feat_dict.get('inter_arrival_mean', 0.0),
                feat_dict.get('inter_arrival_std', 0.0),
                feat_dict.get('size_mean', 0.0),
                feat_dict.get('size_std', 0.0),
                feat_dict.get('entropy', 0.0),
            ])
        else:
            features.extend([0.0] * 5)
        
        # Pad or truncate to fixed size
        target_size = 32
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _create_node_features(
        self,
        node_map: Dict[str, int],
        flows: List[Dict]
    ) -> torch.Tensor:
        """
        Create node features from flow statistics.
        
        Args:
            node_map: IP to node ID mapping
            flows: List of flows
            
        Returns:
            Node feature tensor
        """
        num_nodes = len(node_map)
        node_feat_dim = 64
        
        # Aggregate statistics per node
        node_stats = defaultdict(lambda: {
            'out_degree': 0,
            'in_degree': 0,
            'total_bytes_out': 0,
            'total_bytes_in': 0,
            'unique_destinations': set(),
            'unique_sources': set(),
        })
        
        for flow in flows:
            src_ip = flow['src_ip']
            dst_ip = flow['dst_ip']
            bytes_transferred = flow.get('total_bytes', 0)
            
            if src_ip in node_map:
                node_stats[src_ip]['out_degree'] += 1
                node_stats[src_ip]['total_bytes_out'] += bytes_transferred
                node_stats[src_ip]['unique_destinations'].add(dst_ip)
            
            if dst_ip in node_map:
                node_stats[dst_ip]['in_degree'] += 1
                node_stats[dst_ip]['total_bytes_in'] += bytes_transferred
                node_stats[dst_ip]['unique_sources'].add(src_ip)
        
        # Create feature matrix
        features = []
        for ip in sorted(node_map.keys()):
            stats = node_stats[ip]
            feat = [
                float(stats['out_degree']),
                float(stats['in_degree']),
                float(stats['total_bytes_out']),
                float(stats['total_bytes_in']),
                float(len(stats['unique_destinations'])),
                float(len(stats['unique_sources'])),
            ]
            
            # Pad to target dimension
            while len(feat) < node_feat_dim:
                feat.append(0.0)
            
            features.append(feat[:node_feat_dim])
        
        return torch.tensor(features, dtype=torch.float)


def aggregate_flows_by_time_window(
    packets: List[Dict],
    time_window: float = 60.0
) -> List[Dict]:
    """
    Aggregate packets into flows within time windows.
    
    Args:
        packets: List of packet dictionaries
        time_window: Time window size (seconds)
        
    Returns:
        List of aggregated flows
    """
    if not packets:
        return []
    
    # Sort by timestamp
    sorted_packets = sorted(packets, key=lambda x: x['timestamp'])
    
    # Group packets into flows
    flows = defaultdict(lambda: {
        'packets': [],
        'src_ip': None,
        'dst_ip': None,
        'src_port': None,
        'dst_port': None,
        'protocol': None,
    })
    
    for packet in sorted_packets:
        # Create flow key
        flow_key = (
            packet['src_ip'],
            packet['dst_ip'],
            packet.get('src_port', 0),
            packet.get('dst_port', 0),
            packet.get('protocol', 0)
        )
        
        flow = flows[flow_key]
        flow['packets'].append(packet)
        flow['src_ip'] = packet['src_ip']
        flow['dst_ip'] = packet['dst_ip']
        flow['src_port'] = packet.get('src_port', 0)
        flow['dst_port'] = packet.get('dst_port', 0)
        flow['protocol'] = packet.get('protocol', 0)
    
    # Convert to flow dictionaries
    result = []
    for flow_key, flow_data in flows.items():
        packets = flow_data['packets']
        timestamps = [p['timestamp'] for p in packets]
        
        flow = {
            'src_ip': flow_data['src_ip'],
            'dst_ip': flow_data['dst_ip'],
            'src_port': flow_data['src_port'],
            'dst_port': flow_data['dst_port'],
            'protocol': flow_data['protocol'],
            'timestamp': min(timestamps),
            'packet_count': len(packets),
            'total_bytes': sum(p.get('size', 0) for p in packets),
            'duration': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0,
            'packets': packets,
        }
        
        result.append(flow)
    
    return result


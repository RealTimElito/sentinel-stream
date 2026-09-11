"""Graph construction from network flows."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import TemporalData

logger = logging.getLogger(__name__)

# Re-export for callers that historically imported TemporalData from here.
__all__ = [
    "NetworkGraphBuilder",
    "TemporalData",
    "aggregate_flows_by_time_window",
]


class NetworkGraphBuilder:
    """Build temporal graphs from network flow data."""

    def __init__(
        self,
        time_window: float = 60.0,
        min_flow_duration: float = 0.1,
        max_flow_duration: float = 3600.0,
    ):
        self.time_window = time_window
        self.min_flow_duration = min_flow_duration
        self.max_flow_duration = max_flow_duration

    def ip_to_node_id(self, ip: str, node_map: Dict[str, int]) -> int:
        if ip not in node_map:
            node_map[ip] = len(node_map)
        return node_map[ip]

    def build_temporal_graph(
        self,
        flows: List[Dict],
        node_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> TemporalData:
        if not flows:
            raise ValueError("Empty flow list")

        sorted_flows = sorted(flows, key=lambda x: x["timestamp"])

        node_map: Dict[str, int] = {}
        for flow in sorted_flows:
            self.ip_to_node_id(str(flow["src_ip"]), node_map)
            self.ip_to_node_id(str(flow["dst_ip"]), node_map)

        src_nodes = []
        dst_nodes = []
        timestamps = []
        edge_features = []

        for flow in sorted_flows:
            src_nodes.append(self.ip_to_node_id(str(flow["src_ip"]), node_map))
            dst_nodes.append(self.ip_to_node_id(str(flow["dst_ip"]), node_map))
            timestamps.append(float(flow["timestamp"]))
            edge_features.append(self._extract_edge_features(flow))

        src = torch.tensor(src_nodes, dtype=torch.long)
        dst = torch.tensor(dst_nodes, dtype=torch.long)
        t = torch.tensor(timestamps, dtype=torch.float)
        edge_attr = torch.tensor(np.asarray(edge_features, dtype=np.float32))

        if node_features is None:
            x = self._create_node_features(node_map, sorted_flows)
        else:
            x = torch.tensor(
                [
                    node_features.get(ip, np.zeros(64, dtype=np.float32))
                    for ip in sorted(node_map.keys(), key=lambda k: node_map[k])
                ],
                dtype=torch.float,
            )

        # Attach mapping for dashboard / debugging use.
        data = TemporalData(src=src, dst=dst, t=t, edge_attr=edge_attr, x=x)
        data.node_map = dict(node_map)
        return data

    def _extract_edge_features(self, flow: Dict) -> np.ndarray:
        features = [
            float(flow.get("packet_count", 0)),
            float(flow.get("total_bytes", flow.get("packet_size", flow.get("size", 0)))),
            float(flow.get("duration", 0.0)),
            float(flow.get("protocol", 0)),
            float(flow.get("src_port", 0)),
            float(flow.get("dst_port", 0)),
        ]

        feat_dict = flow.get("features") or {}
        features.extend(
            [
                float(feat_dict.get("inter_arrival_mean", 0.0)),
                float(feat_dict.get("inter_arrival_std", 0.0)),
                float(feat_dict.get("size_mean", 0.0)),
                float(feat_dict.get("size_std", 0.0)),
                float(feat_dict.get("entropy", 0.0)),
            ]
        )

        target_size = 32
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        return np.asarray(features, dtype=np.float32)

    def _create_node_features(self, node_map: Dict[str, int], flows: List[Dict]) -> torch.Tensor:
        node_feat_dim = 64
        node_stats = defaultdict(
            lambda: {
                "out_degree": 0,
                "in_degree": 0,
                "total_bytes_out": 0.0,
                "total_bytes_in": 0.0,
                "unique_destinations": set(),
                "unique_sources": set(),
            }
        )

        for flow in flows:
            src_ip = str(flow["src_ip"])
            dst_ip = str(flow["dst_ip"])
            bytes_transferred = float(
                flow.get("total_bytes", flow.get("packet_size", flow.get("size", 0)))
            )

            node_stats[src_ip]["out_degree"] += 1
            node_stats[src_ip]["total_bytes_out"] += bytes_transferred
            node_stats[src_ip]["unique_destinations"].add(dst_ip)

            node_stats[dst_ip]["in_degree"] += 1
            node_stats[dst_ip]["total_bytes_in"] += bytes_transferred
            node_stats[dst_ip]["unique_sources"].add(src_ip)

        features = []
        for ip, _ in sorted(node_map.items(), key=lambda item: item[1]):
            stats = node_stats[ip]
            feat = [
                float(stats["out_degree"]),
                float(stats["in_degree"]),
                float(stats["total_bytes_out"]),
                float(stats["total_bytes_in"]),
                float(len(stats["unique_destinations"])),
                float(len(stats["unique_sources"])),
            ]
            feat.extend([0.0] * (node_feat_dim - len(feat)))
            features.append(feat[:node_feat_dim])

        return torch.tensor(features, dtype=torch.float)


def aggregate_flows_by_time_window(packets: List[Dict], time_window: float = 60.0) -> List[Dict]:
    """Aggregate packets into flows (5-tuple), ignoring empty input."""
    del time_window  # reserved for future windowed splitting
    if not packets:
        return []

    sorted_packets = sorted(packets, key=lambda x: x["timestamp"])
    flows = defaultdict(
        lambda: {
            "packets": [],
            "src_ip": None,
            "dst_ip": None,
            "src_port": None,
            "dst_port": None,
            "protocol": None,
        }
    )

    for packet in sorted_packets:
        flow_key = (
            packet["src_ip"],
            packet["dst_ip"],
            packet.get("src_port", 0),
            packet.get("dst_port", 0),
            packet.get("protocol", 0),
        )
        flow = flows[flow_key]
        flow["packets"].append(packet)
        flow["src_ip"] = packet["src_ip"]
        flow["dst_ip"] = packet["dst_ip"]
        flow["src_port"] = packet.get("src_port", 0)
        flow["dst_port"] = packet.get("dst_port", 0)
        flow["protocol"] = packet.get("protocol", 0)

    result = []
    for flow_data in flows.values():
        pkts = flow_data["packets"]
        timestamps = [p["timestamp"] for p in pkts]
        sizes = [p.get("size", p.get("packet_size", 0)) for p in pkts]
        result.append(
            {
                "src_ip": flow_data["src_ip"],
                "dst_ip": flow_data["dst_ip"],
                "src_port": flow_data["src_port"],
                "dst_port": flow_data["dst_port"],
                "protocol": flow_data["protocol"],
                "timestamp": min(timestamps),
                "packet_count": len(pkts),
                "total_bytes": sum(sizes),
                "duration": (max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0),
                "packets": pkts,
            }
        )
    return result

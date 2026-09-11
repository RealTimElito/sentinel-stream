"""Tests for graph construction and TGN forward pass."""

import torch

from src.ml.explainability import TGNExplainer
from src.ml.graph_construction import (
    NetworkGraphBuilder,
    aggregate_flows_by_time_window,
)
from src.ml.tgn_model import TemporalGraphNetwork
from src.utils.feature_engineering import extract_flow_features


def _sample_packets():
    return [
        {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1111,
            "dst_port": 80,
            "protocol": 6,
            "timestamp": 0.0,
            "size": 64,
        },
        {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1111,
            "dst_port": 80,
            "protocol": 6,
            "timestamp": 0.5,
            "size": 128,
        },
        {
            "src_ip": "10.0.0.3",
            "dst_ip": "10.0.0.4",
            "src_port": 2222,
            "dst_port": 443,
            "protocol": 6,
            "timestamp": 1.0,
            "size": 256,
        },
    ]


def test_aggregate_flows_by_time_window():
    flows = aggregate_flows_by_time_window(_sample_packets())
    assert len(flows) == 2
    assert flows[0]["packet_count"] == 2
    assert flows[0]["total_bytes"] == 192


def test_build_temporal_graph_and_forward():
    packets = _sample_packets()
    flows = aggregate_flows_by_time_window(packets)
    for flow in flows:
        flow["features"] = extract_flow_features(flow["packets"])

    builder = NetworkGraphBuilder()
    graph = builder.build_temporal_graph(flows)
    assert graph.src.numel() == 2
    assert graph.x.shape[1] == 64
    assert graph.edge_attr.shape[1] == 32

    model = TemporalGraphNetwork()
    model.eval()
    with torch.no_grad():
        scores, embeddings = model(graph, return_embeddings=True)
    assert scores.shape[0] == graph.src.shape[0]
    assert embeddings is not None
    assert len(embeddings) == graph.src.shape[0]


def test_predict_next_edge():
    packets = _sample_packets()
    flows = aggregate_flows_by_time_window(packets)
    for flow in flows:
        flow["features"] = extract_flow_features(flow["packets"])
    graph = NetworkGraphBuilder().build_temporal_graph(flows)
    model = TemporalGraphNetwork()
    preds = model.predict_next_edge(graph)
    assert preds.shape[0] == graph.src.shape[0]


def test_explainer_graph():
    packets = _sample_packets()
    flows = aggregate_flows_by_time_window(packets)
    for flow in flows:
        flow["features"] = extract_flow_features(flow["packets"])
    graph = NetworkGraphBuilder().build_temporal_graph(flows)
    model = TemporalGraphNetwork()
    explanation = TGNExplainer(model).explain_graph(graph)
    assert "node_importance" in explanation
    assert "edge_importance" in explanation
    assert len(explanation["edge_importance"]) == graph.src.shape[0]

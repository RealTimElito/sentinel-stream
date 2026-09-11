"""Tests for feature engineering."""

import numpy as np

from src.utils.feature_engineering import (
    calculate_flow_entropy,
    calculate_inter_arrival_times,
    calculate_packet_size_sequence,
    extract_flow_features,
)


def test_calculate_inter_arrival_times():
    timestamps = [0.0, 1.0, 2.5, 3.0]
    inter_arrivals = calculate_inter_arrival_times(timestamps)
    assert len(inter_arrivals) == 3
    assert np.isclose(inter_arrivals[0], 1.0)
    assert np.isclose(inter_arrivals[1], 1.5)
    assert np.isclose(inter_arrivals[2], 0.5)


def test_calculate_packet_size_sequence():
    sizes = [64, 128, 256, 512, 1024]
    features = calculate_packet_size_sequence(sizes)
    assert len(features) == 7
    assert features[0] == np.mean(sizes)
    assert features[2] == np.min(sizes)
    assert features[3] == np.max(sizes)


def test_calculate_flow_entropy():
    sizes = [64, 64, 128, 128, 256]
    entropy = calculate_flow_entropy(sizes)
    assert entropy >= 0
    assert entropy <= 10


def test_extract_flow_features():
    packets = [
        {"timestamp": 0.0, "size": 64},
        {"timestamp": 1.0, "size": 128},
        {"timestamp": 2.0, "size": 256},
    ]
    features = extract_flow_features(packets)
    assert "inter_arrival_mean" in features
    assert "size_mean" in features
    assert "entropy" in features
    assert "packet_count" in features
    assert features["packet_count"] == 3

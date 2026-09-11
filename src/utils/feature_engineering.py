"""Feature engineering for network flows."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def calculate_inter_arrival_times(timestamps: List[float]) -> np.ndarray:
    if len(timestamps) < 2:
        return np.array([])
    timestamps_sorted = sorted(timestamps)
    return np.diff(timestamps_sorted)


def calculate_packet_size_sequence(packet_sizes: List[int]) -> np.ndarray:
    if not packet_sizes:
        return np.array([])
    sizes = np.array(packet_sizes)
    features = {
        "mean": np.mean(sizes),
        "std": np.std(sizes),
        "min": np.min(sizes),
        "max": np.max(sizes),
        "median": np.median(sizes),
        "percentile_25": np.percentile(sizes, 25),
        "percentile_75": np.percentile(sizes, 75),
    }
    return np.array(list(features.values()))


def calculate_flow_entropy(packet_sizes: List[int], num_bins: int = 10) -> float:
    if not packet_sizes or len(packet_sizes) < 2:
        return 0.0
    hist, _ = np.histogram(packet_sizes, bins=num_bins)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def extract_flow_features(packets: List[Dict], time_window: float = 60.0) -> Dict[str, float]:
    if not packets:
        return {}

    timestamps = [p["timestamp"] for p in packets]
    sizes = [p["size"] for p in packets]

    if timestamps:
        start_time = min(timestamps)
        filtered_packets = [p for p in packets if p["timestamp"] <= start_time + time_window]
        if filtered_packets:
            timestamps = [p["timestamp"] for p in filtered_packets]
            sizes = [p["size"] for p in filtered_packets]

    features: Dict[str, float] = {}
    inter_arrivals = calculate_inter_arrival_times(timestamps)
    if len(inter_arrivals) > 0:
        features["inter_arrival_mean"] = float(np.mean(inter_arrivals))
        features["inter_arrival_std"] = float(np.std(inter_arrivals))
        features["inter_arrival_min"] = float(np.min(inter_arrivals))
        features["inter_arrival_max"] = float(np.max(inter_arrivals))
    else:
        features["inter_arrival_mean"] = 0.0
        features["inter_arrival_std"] = 0.0
        features["inter_arrival_min"] = 0.0
        features["inter_arrival_max"] = 0.0

    size_features = calculate_packet_size_sequence(sizes)
    if len(size_features) > 0:
        features["size_mean"] = float(size_features[0])
        features["size_std"] = float(size_features[1])
        features["size_min"] = float(size_features[2])
        features["size_max"] = float(size_features[3])
        features["size_median"] = float(size_features[4])
        features["size_p25"] = float(size_features[5])
        features["size_p75"] = float(size_features[6])

    features["entropy"] = calculate_flow_entropy(sizes)
    features["packet_count"] = float(len(packets))
    features["total_bytes"] = float(sum(sizes))
    features["duration"] = float(max(timestamps) - min(timestamps)) if timestamps else 0.0

    protocols = [p.get("protocol", 0) for p in packets]
    if protocols:
        unique_protocols = len(set(protocols))
        features["unique_protocols"] = float(unique_protocols)
        features["protocol_diversity"] = float(unique_protocols / len(protocols))
    else:
        features["unique_protocols"] = 0.0
        features["protocol_diversity"] = 0.0

    return features


def normalize_features(
    features: Dict[str, float],
    feature_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for name, values in features.items():
        if feature_stats and name in feature_stats:
            mean, std = feature_stats[name]
            normalized[name] = (values - mean) / std if std > 0 else values - mean
        else:
            normalized[name] = values
    return normalized

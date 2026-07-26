"""Feature engineering for network flows."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math


def calculate_inter_arrival_times(timestamps: List[float]) -> np.ndarray:
    """
    Calculate inter-arrival times between packets.
    
    Args:
        timestamps: List of packet timestamps
        
    Returns:
        Array of inter-arrival times
    """
    if len(timestamps) < 2:
        return np.array([])
    
    timestamps_sorted = sorted(timestamps)
    inter_arrivals = np.diff(timestamps_sorted)
    return inter_arrivals


def calculate_packet_size_sequence(packet_sizes: List[int]) -> np.ndarray:
    """
    Extract packet size sequence features.
    
    Args:
        packet_sizes: List of packet sizes
        
    Returns:
        Array of packet size features
    """
    if not packet_sizes:
        return np.array([])
    
    sizes = np.array(packet_sizes)
    
    # Calculate statistics
    features = {
        'mean': np.mean(sizes),
        'std': np.std(sizes),
        'min': np.min(sizes),
        'max': np.max(sizes),
        'median': np.median(sizes),
        'percentile_25': np.percentile(sizes, 25),
        'percentile_75': np.percentile(sizes, 75),
    }
    
    return np.array(list(features.values()))


def calculate_flow_entropy(
    packet_sizes: List[int],
    num_bins: int = 10
) -> float:
    """
    Calculate entropy of packet size distribution.
    
    Args:
        packet_sizes: List of packet sizes
        num_bins: Number of bins for histogram
        
    Returns:
        Entropy value
    """
    if not packet_sizes or len(packet_sizes) < 2:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(packet_sizes, bins=num_bins)
    
    # Normalize to probabilities
    prob = hist / np.sum(hist)
    
    # Remove zeros
    prob = prob[prob > 0]
    
    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob))
    
    return float(entropy)


def extract_flow_features(
    packets: List[Dict],
    time_window: float = 60.0
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive features from a flow.
    
    Args:
        packets: List of packet dictionaries with 'timestamp', 'size', etc.
        time_window: Time window for flow aggregation (seconds)
        
    Returns:
        Dictionary of feature arrays
    """
    if not packets:
        return {}
    
    timestamps = [p['timestamp'] for p in packets]
    sizes = [p['size'] for p in packets]
    
    # Filter by time window
    if timestamps:
        start_time = min(timestamps)
        filtered_packets = [
            p for p in packets
            if p['timestamp'] <= start_time + time_window
        ]
        
        if filtered_packets:
            timestamps = [p['timestamp'] for p in filtered_packets]
            sizes = [p['size'] for p in filtered_packets]
    
    features = {}
    
    # Inter-arrival times
    inter_arrivals = calculate_inter_arrival_times(timestamps)
    if len(inter_arrivals) > 0:
        features['inter_arrival_mean'] = np.mean(inter_arrivals)
        features['inter_arrival_std'] = np.std(inter_arrivals)
        features['inter_arrival_min'] = np.min(inter_arrivals)
        features['inter_arrival_max'] = np.max(inter_arrivals)
    else:
        features['inter_arrival_mean'] = 0.0
        features['inter_arrival_std'] = 0.0
        features['inter_arrival_min'] = 0.0
        features['inter_arrival_max'] = 0.0
    
    # Packet size features
    size_features = calculate_packet_size_sequence(sizes)
    if len(size_features) > 0:
        features['size_mean'] = size_features[0]
        features['size_std'] = size_features[1]
        features['size_min'] = size_features[2]
        features['size_max'] = size_features[3]
        features['size_median'] = size_features[4]
        features['size_p25'] = size_features[5]
        features['size_p75'] = size_features[6]
    
    # Flow entropy
    features['entropy'] = calculate_flow_entropy(sizes)
    
    # Flow statistics
    features['packet_count'] = len(packets)
    features['total_bytes'] = sum(sizes)
    features['duration'] = max(timestamps) - min(timestamps) if timestamps else 0.0
    
    # Protocol distribution
    protocols = [p.get('protocol', 0) for p in packets]
    if protocols:
        unique_protocols = len(set(protocols))
        features['unique_protocols'] = unique_protocols
        features['protocol_diversity'] = unique_protocols / len(protocols)
    else:
        features['unique_protocols'] = 0
        features['protocol_diversity'] = 0.0
    
    return features


def normalize_features(
    features: Dict[str, np.ndarray],
    feature_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Dictionary of feature arrays
        feature_stats: Dictionary mapping feature names to (mean, std) tuples
        
    Returns:
        Normalized features
    """
    normalized = {}
    
    for name, values in features.items():
        if feature_stats and name in feature_stats:
            mean, std = feature_stats[name]
            if std > 0:
                normalized[name] = (values - mean) / std
            else:
                normalized[name] = values - mean
        else:
            # Use sample statistics
            if isinstance(values, np.ndarray) and len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    normalized[name] = (values - mean) / std
                else:
                    normalized[name] = values - mean
            else:
                normalized[name] = values
    
    return normalized


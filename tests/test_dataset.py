"""Tests for dataset mapping helpers."""

import pandas as pd

from scripts.prepare_dataset import map_cic_ids2017_features, preprocess_data


def test_map_cic_ids2017_features_without_ips():
    df = pd.DataFrame(
        {
            "Destination Port": [80, 443],
            "Flow Duration": [10, 20],
            "Total Length of Fwd Packets": [100, 200],
            "Protocol": [6, 17],
            "Label": ["BENIGN", "DDoS"],
        }
    )
    mapped = map_cic_ids2017_features(df)
    assert list(mapped["dst_port"]) == [80, 443]
    assert list(mapped["packet_size"]) == [100, 200]
    assert list(mapped["label"]) == [0, 1]
    assert "src_ip" in mapped.columns
    assert "dst_ip" in mapped.columns


def test_preprocess_creates_port_based_nodes():
    df = pd.DataFrame(
        {
            "src_ip": ["0.0.0.0", "0.0.0.0"],
            "dst_ip": ["0.0.0.0", "0.0.0.0"],
            "src_port": [1111, 2222],
            "dst_port": [80, 443],
            "protocol": [6, 6],
            "timestamp": [0.0, 1.0],
            "packet_size": [64, 128],
            "label": [0, 1],
        }
    )
    processed = preprocess_data(df)
    assert len(processed) == 2
    assert processed["src_ip"].str.startswith("port_").all()

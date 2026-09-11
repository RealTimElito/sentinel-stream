#!/usr/bin/env python3
"""Prepare dataset for training."""

import argparse
import glob
import logging
import os
import tarfile
import urllib.request
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """

    def reporthook(blocknum, blocksize, totalsize):
        if totalsize > 0:
            percent = min(100, (blocknum * blocksize * 100) // totalsize)
            print(f"\r{desc}: {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # New line after progress
        logger.info(f"Downloaded to {dest_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def download_network_intrusion_dataset(data_path: str) -> bool:
    """
    Download network-intrusion-dataset from Kaggle via curl (no auth required).

    Args:
        data_path: Path to save dataset

    Returns:
        True if download successful, False otherwise
    """
    import subprocess

    url = "https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset"
    os.makedirs(data_path, exist_ok=True)
    filename = os.path.join(data_path, "network-intrusion-dataset.zip")

    logger.info("Downloading network-intrusion-dataset (no authentication required)...")

    cmd = ["curl", "-L", "--progress-bar", "-o", filename, url]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded to {filename}")

        # Extract if zip file
        if filename.endswith(".zip"):
            logger.info("Extracting archive...")
            import zipfile

            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(data_path)
            logger.info("Extraction complete!")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("curl not found. Install with: sudo apt-get install curl")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def download_cic_ids2017(data_path: str, force: bool = False) -> bool:
    """
    Download CIC-IDS2017 dataset.

    Args:
        data_path: Path to save dataset
        force: Force re-download even if files exist

    Returns:
        True if download successful, False otherwise
    """
    os.makedirs(data_path, exist_ok=True)

    # Check if files already exist
    csv_files = find_csv_files(data_path)
    if csv_files and not force:
        logger.info(f"Dataset files already exist in {data_path}")
        return True

    logger.info("Attempting to download CIC-IDS2017 dataset...")

    # Try Kaggle API first (most reliable)
    try:
        import kaggle

        logger.info("Found Kaggle API, attempting download...")
        try:
            kaggle.api.dataset_download_files("cicdataset/cicids2017", path=data_path, unzip=True)
            logger.info("✓ Download successful via Kaggle!")
            return True
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
            logger.info("You may need to set up Kaggle credentials:")
            logger.info("1. Get API token from https://www.kaggle.com/account")
            logger.info("2. Place kaggle.json in ~/.kaggle/")
    except ImportError:
        logger.info("Kaggle API not available. Install with: pip install kaggle")

    # Try alternative dataset via curl
    logger.info("Trying alternative dataset: network-intrusion-dataset...")
    if download_network_intrusion_dataset(data_path):
        logger.info("✓ Alternative dataset downloaded successfully!")
        return True

    # Fallback: Provide instructions
    logger.warning("Automatic download not available. Options:")
    logger.info("")
    logger.info("Option 1: Use Kaggle (recommended)")
    logger.info("  pip install kaggle")
    logger.info("  python scripts/download_dataset.py --method kaggle")
    logger.info("")
    logger.info("Option 2: Download alternative dataset via curl")
    logger.info("  curl -L -o ~/Downloads/network-intrusion-dataset.zip \\")
    logger.info(
        "    https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset"
    )
    logger.info("  python scripts/download_dataset.py --method network-intrusion")
    logger.info("")
    logger.info("Option 3: Manual download")
    logger.info("  1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
    logger.info("  2. Register and download")
    logger.info("  3. Extract CSV files to: ./data/raw/")
    logger.info("")
    logger.info("Option 4: Use placeholder data for testing")
    logger.info("  (Will be used automatically if no files found)")

    return False


def extract_archive(archive_path: str, extract_to: str):
    """
    Extract archive file (zip or tar).

    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    logger.info("Extraction complete")


def find_csv_files(data_path: str) -> List[str]:
    """
    Find all CSV files in the dataset directory.

    Args:
        data_path: Path to dataset directory

    Returns:
        List of CSV file paths
    """
    csv_files = []

    # Look for CSV files in the directory
    patterns = [
        os.path.join(data_path, "*.csv"),
        os.path.join(data_path, "**", "*.csv"),
        os.path.join(data_path, "MachineLearningCSV", "*.csv"),
    ]

    for pattern in patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates and sort
    csv_files = sorted(list(set(csv_files)))

    return csv_files


def map_cic_ids2017_features(df: pd.DataFrame, preserve_all_features: bool = False) -> pd.DataFrame:
    """
    Map CIC-IDS2017 features to our standard format.

    CIC-IDS2017 column names vary, so we try multiple possible names.
    The MachineLearningCSV release often omits Source/Destination IP columns;
    in that case ports are used later as graph node identifiers.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    column_mapping = {}

    for col in ["Source IP", "Src IP", "SourceIP", "src_ip"]:
        if col in df.columns:
            column_mapping[col] = "src_ip"
            break

    for col in ["Destination IP", "Dst IP", "DestinationIP", "dst_ip"]:
        if col in df.columns:
            column_mapping[col] = "dst_ip"
            break

    for col in df.columns:
        col_clean = col.strip()
        if col_clean in ["Source Port", "Src Port", "SourcePort", "src_port"]:
            column_mapping[col] = "src_port"
            break

    for col in df.columns:
        col_clean = col.strip()
        if col_clean in [
            "Destination Port",
            "Dst Port",
            "DestinationPort",
            "dst_port",
        ]:
            column_mapping[col] = "dst_port"
            break

    for col in df.columns:
        if col.strip() in ["Protocol", "protocol"]:
            column_mapping[col] = "protocol"
            break

    # Prefer true timestamps; fall back to flow duration as a relative clock.
    for preferred in ["Timestamp", "timestamp"]:
        if preferred in df.columns:
            column_mapping[preferred] = "timestamp"
            break
    else:
        for col in df.columns:
            if col.strip() in ["Flow Duration", "FlowDuration"]:
                column_mapping[col] = "timestamp"
                break

    for col in df.columns:
        col_clean = col.strip()
        if col_clean in [
            "Total Length of Fwd Packets",
            "Total Length",
            "packet_size",
            "Packet Length Mean",
            "Average Packet Size",
        ]:
            column_mapping[col] = "packet_size"
            break

    for col in df.columns:
        if col.strip() in ["Label", "label", "Attack", "attack"]:
            column_mapping[col] = "label"
            break

    df_mapped = df.rename(columns=column_mapping)

    if preserve_all_features:
        result = df_mapped.copy()
        cols_to_remove = [
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "timestamp",
            "packet_size",
            "label",
        ]
        for col in cols_to_remove:
            if col in result.columns:
                result = result.drop(columns=[col])
    else:
        result = pd.DataFrame(index=df.index)

    # Required columns with defaults
    num_rows = len(df)

    if "src_ip" in df_mapped.columns:
        result["src_ip"] = df_mapped["src_ip"].fillna("0.0.0.0")
    else:
        result["src_ip"] = pd.Series(["0.0.0.0"] * num_rows, index=df.index)

    if "dst_ip" in df_mapped.columns:
        result["dst_ip"] = df_mapped["dst_ip"].fillna("0.0.0.0")
    else:
        result["dst_ip"] = pd.Series(["0.0.0.0"] * num_rows, index=df.index)

    if "src_port" in df_mapped.columns:
        result["src_port"] = (
            pd.to_numeric(df_mapped["src_port"], errors="coerce").fillna(0).astype(int)
        )
    else:
        result["src_port"] = pd.Series([0] * num_rows, index=df.index, dtype=int)

    if "dst_port" in df_mapped.columns:
        result["dst_port"] = (
            pd.to_numeric(df_mapped["dst_port"], errors="coerce").fillna(0).astype(int)
        )
    else:
        result["dst_port"] = pd.Series([0] * num_rows, index=df.index, dtype=int)

    # Protocol mapping (TCP=6, UDP=17, ICMP=1)
    if "protocol" in df_mapped.columns:
        protocol = df_mapped["protocol"]
        # Check if it's string type
        if protocol.dtype == "object" or (len(protocol) > 0 and isinstance(protocol.iloc[0], str)):
            protocol_map = {"TCP": 6, "UDP": 17, "ICMP": 1, "tcp": 6, "udp": 17, "icmp": 1}
            result["protocol"] = (
                protocol.astype(str).str.upper().map(protocol_map).fillna(6).astype(int)
            )
        else:
            result["protocol"] = pd.to_numeric(protocol, errors="coerce").fillna(6).astype(int)
    else:
        result["protocol"] = pd.Series([6] * num_rows, index=df.index, dtype=int)  # Default to TCP

    # Timestamp (convert to seconds if needed)
    if "timestamp" in df_mapped.columns:
        timestamp = df_mapped["timestamp"]
        # Try to parse as datetime if string
        if timestamp.dtype == "object":
            try:
                timestamp_parsed = pd.to_datetime(timestamp, errors="coerce")
                if timestamp_parsed.notna().any():
                    result["timestamp"] = (
                        (timestamp_parsed - timestamp_parsed.min()).dt.total_seconds().fillna(0)
                    )
                else:
                    result["timestamp"] = pd.Series(
                        np.arange(num_rows), index=df.index, dtype=float
                    )
            except (TypeError, ValueError):
                result["timestamp"] = pd.Series(np.arange(num_rows), index=df.index, dtype=float)
        else:
            result["timestamp"] = pd.to_numeric(timestamp, errors="coerce").fillna(0)
    else:
        result["timestamp"] = pd.Series(np.arange(num_rows), index=df.index, dtype=float)

    # Packet size
    if "packet_size" in df_mapped.columns:
        result["packet_size"] = (
            pd.to_numeric(df_mapped["packet_size"], errors="coerce").fillna(0).astype(int)
        )
    else:
        result["packet_size"] = pd.Series([0] * num_rows, index=df.index, dtype=int)

    # Label (BENIGN=0, attacks=1)
    if "label" in df_mapped.columns:
        label = df_mapped["label"]
        # Convert to string and check if BENIGN
        label_str = label.astype(str).str.upper()
        result["label"] = (label_str != "BENIGN").astype(int)
    else:
        # Default to all benign if no label column
        result["label"] = pd.Series([0] * num_rows, index=df.index, dtype=int)

    return result


def load_cic_ids2017(
    data_path: str, max_files: Optional[int] = None, auto_download: bool = False
) -> pd.DataFrame:
    """
    Load CIC-IDS2017 dataset from CSV files.

    Args:
        data_path: Path to dataset directory
        max_files: Maximum number of CSV files to load (None for all)
        auto_download: Attempt to download dataset if not found

    Returns:
        DataFrame with network flow data
    """
    logger.info(f"Loading CIC-IDS2017 dataset from {data_path}")

    # Find CSV files
    csv_files = find_csv_files(data_path)

    if not csv_files:
        if auto_download:
            logger.info("No CSV files found, attempting download...")
            download_cic_ids2017(data_path)
            # Try again after download attempt
            csv_files = find_csv_files(data_path)

        if not csv_files:
            logger.warning(f"No CSV files found in {data_path}")
            logger.info("Creating placeholder data. To use real CIC-IDS2017 data:")
            logger.info("1. Download from https://www.unb.ca/cic/datasets/ids-2017.html")
            logger.info(
                "   Or use: python scripts/prepare_dataset.py --dataset CIC-IDS2017 --download"
            )
            logger.info("2. Extract CSV files to ./data/raw/")
            logger.info("3. Run this script again")

            # Return placeholder data
            data = {
                "src_ip": ["192.168.1.1"] * 1000,
                "dst_ip": ["192.168.1.2"] * 1000,
                "src_port": np.random.randint(1024, 65535, 1000),
                "dst_port": np.random.randint(1024, 65535, 1000),
                "protocol": np.random.choice([6, 17], 1000),
                "timestamp": np.random.uniform(0, 3600, 1000),
                "packet_size": np.random.randint(64, 1500, 1000),
                "label": np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
            }
            return pd.DataFrame(data)

    logger.info(f"Found {len(csv_files)} CSV file(s)")

    if max_files:
        csv_files = csv_files[:max_files]
        logger.info(f"Loading first {len(csv_files)} file(s)")

    # Load and combine CSV files
    dataframes = []
    for i, csv_file in enumerate(csv_files):
        try:
            logger.info(f"Loading {os.path.basename(csv_file)} ({i+1}/{len(csv_files)})...")

            # Try to read with different encodings
            df = None
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                logger.warning(f"Failed to read {csv_file}, skipping")
                continue

            # Map features
            df_mapped = map_cic_ids2017_features(df)
            dataframes.append(df_mapped)

            logger.info(f"  Loaded {len(df_mapped)} rows")

        except Exception as e:
            logger.warning(f"Error loading {csv_file}: {e}, skipping")
            continue

    if not dataframes:
        raise ValueError("No data could be loaded from CSV files")

    # Combine all dataframes
    logger.info("Combining dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    logger.info(f"Total rows loaded: {len(combined_df)}")
    logger.info(f"Anomaly rate: {combined_df['label'].mean()*100:.2f}%")

    return combined_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset.

    Args:
        df: Raw dataframe

    Returns:
        Preprocessed dataframe
    """
    logger.info("Preprocessing data...")

    # Remove rows with invalid IP addresses (if IP columns exist)
    initial_len = len(df)

    if "src_ip" in df.columns and "dst_ip" in df.columns:
        # Convert IPs to string and check validity
        df["src_ip"] = df["src_ip"].astype(str)
        df["dst_ip"] = df["dst_ip"].astype(str)

        # Check if all IPs are '0.0.0.0' (meaning they're synthetic defaults)
        all_zero_ips = ((df["src_ip"] == "0.0.0.0") & (df["dst_ip"] == "0.0.0.0")).all()

        if all_zero_ips:
            # All IPs are defaults - use flow-based identifiers instead of synthetic IPs
            # This preserves the actual flow data from the dataset
            logger.info(
                "All IPs are default values. Using flow-based identifiers (preserving raw data)..."
            )

            # Create unique flow identifiers from actual data
            if "src_port" in df.columns and "dst_port" in df.columns:
                unique_src_ports = df["src_port"].unique()
                unique_dst_ports = df["dst_port"].unique()
                df["src_ip"] = "port_" + df["src_port"].astype(str)
                df["dst_ip"] = "port_" + df["dst_port"].astype(str)
                logger.info(
                    "Using %s unique source ports and %s unique destination ports as nodes",
                    len(unique_src_ports),
                    len(unique_dst_ports),
                )
                logger.info("This preserves the actual flow characteristics from the dataset")
            else:
                # Fallback: use row indices as flow identifiers
                logger.warning("No port columns found. Using row-based identifiers...")
                df["src_ip"] = "flow_" + df.index.astype(str) + "_src"
                df["dst_ip"] = "flow_" + df.index.astype(str) + "_dst"
        else:
            # Filter out invalid IPs (NaN, 'nan', '0.0.0.0', empty strings)
            valid_mask = (
                df["src_ip"].notna()
                & df["dst_ip"].notna()
                & (df["src_ip"] != "nan")
                & (df["dst_ip"] != "nan")
                & (df["src_ip"] != "0.0.0.0")
                & (df["dst_ip"] != "0.0.0.0")
                & (df["src_ip"] != "")
                & (df["dst_ip"] != "")
            )

            df = df[valid_mask]

            removed = initial_len - len(df)
            if removed > 0:
                pct = removed / initial_len * 100
                logger.info(
                    "Removed %s rows with invalid IP addresses (%.1f%%)",
                    removed,
                    pct,
                )
    else:
        # If no IP columns, use flow-based identifiers from actual data
        logger.info(
            "No IP address columns found. Using flow-based identifiers (preserving raw data)..."
        )

        # Create identifiers from actual flow data
        if "src_port" in df.columns and "dst_port" in df.columns:
            # Use ports as node identifiers - preserves actual port information
            unique_src_ports = df["src_port"].unique()
            unique_dst_ports = df["dst_port"].unique()

            df["src_ip"] = "port_" + df["src_port"].astype(str)
            df["dst_ip"] = "port_" + df["dst_port"].astype(str)

            logger.info(
                "Using %s unique source ports and %s unique destination ports as nodes",
                len(unique_src_ports),
                len(unique_dst_ports),
            )
            logger.info("This preserves the actual flow characteristics from the dataset")
        else:
            # Fallback: use row indices as flow identifiers
            logger.warning("No port columns found. Using row-based identifiers...")
            df["src_ip"] = "flow_" + df.index.astype(str) + "_src"
            df["dst_ip"] = "flow_" + df.index.astype(str) + "_dst"

    # Ensure ports are in valid range
    df["src_port"] = df["src_port"].clip(0, 65535)
    df["dst_port"] = df["dst_port"].clip(0, 65535)

    # Ensure protocol is valid (TCP=6, UDP=17, ICMP=1)
    df["protocol"] = df["protocol"].replace({0: 6})  # Default to TCP
    df["protocol"] = df["protocol"].clip(1, 255)

    # Ensure packet size is reasonable
    df["packet_size"] = df["packet_size"].clip(0, 65535)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(
        subset=["src_ip", "dst_ip", "src_port", "dst_port", "timestamp"], keep="first"
    )
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")

    logger.info(f"Final dataset size: {len(df)} rows")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIC-IDS2017",
        choices=["CIC-IDS2017", "CSE-CIC-IDS2018"],
        help="Dataset to use",
    )
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument(
        "--output", type=str, default="./data/processed", help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of CSV files to load (for testing)",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to raw CSV files (defaults to data_path/raw)",
    )
    parser.add_argument(
        "--download", action="store_true", help="Attempt to download dataset if not found"
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine raw data path
    if args.raw_data_path:
        raw_path = args.raw_data_path
    else:
        raw_path = os.path.join(args.data_path, "raw")

    # Download if requested
    if args.download or args.force_download:
        download_cic_ids2017(raw_path, force=args.force_download)

    # Load dataset
    if args.dataset == "CIC-IDS2017":
        df = load_cic_ids2017(raw_path, max_files=args.max_files, auto_download=args.download)
    else:
        logger.error(f"Dataset {args.dataset} not yet implemented")
        return

    # Preprocess
    df = preprocess_data(df)

    # Save processed data
    output_path = os.path.join(args.output, f"{args.dataset.lower()}_processed.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download CIC-IDS2017 dataset using various methods."""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_via_kaggle(data_path: str):
    """Download dataset via Kaggle API."""
    logger.info("Attempting to download via Kaggle...")

    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        logger.info("Then set up credentials: https://www.kaggle.com/docs/api")
        return False

    try:
        os.makedirs(data_path, exist_ok=True)
        kaggle.api.dataset_download_files("cicdataset/cicids2017", path=data_path, unzip=True)
        logger.info("Download successful!")
        return True
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def download_via_wget(url: str, data_path: str):
    """Download dataset via wget."""
    logger.info(f"Attempting to download via wget from {url}...")

    os.makedirs(data_path, exist_ok=True)

    try:
        subprocess.run(["wget", "--continue", "--progress=bar", "-P", data_path, url], check=True)
        logger.info("Download successful!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"wget download failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("wget not found. Install with: sudo apt-get install wget")
        return False


def download_via_curl(url: str, data_path: str):
    """Download dataset via curl."""
    logger.info(f"Attempting to download via curl from {url}...")

    os.makedirs(data_path, exist_ok=True)

    # Determine filename
    if url.endswith(".zip"):
        filename = os.path.join(data_path, os.path.basename(url))
    else:
        filename = os.path.join(data_path, "dataset.zip")

    # Build curl command (no auth needed for this dataset)
    cmd = ["curl", "-L", "--progress-bar", "-o", filename, url]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Download successful! Saved to {filename}")

        # Try to extract if it's a zip file
        if filename.endswith(".zip"):
            logger.info("Extracting archive...")
            import zipfile

            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(data_path)
            logger.info("Extraction complete!")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"curl download failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("curl not found. Install with: sudo apt-get install curl")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def download_network_intrusion_dataset(data_path: str):
    """Download network-intrusion-dataset from Kaggle (no auth required)."""
    url = "https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset"
    logger.info("Downloading network-intrusion-dataset (no authentication required)...")
    return download_via_curl(url, data_path)


def main():
    parser = argparse.ArgumentParser(description="Download network intrusion datasets")
    parser.add_argument(
        "--method",
        type=str,
        choices=["kaggle", "wget", "curl", "network-intrusion", "manual"],
        default="manual",
        help="Download method",
    )
    parser.add_argument("--url", type=str, help="Direct download URL (for wget/curl methods)")
    parser.add_argument("--data-path", type=str, default="./data/raw", help="Path to save dataset")
    args = parser.parse_args()

    logger.info("Network Intrusion Dataset Downloader")
    logger.info("=" * 50)

    if args.method == "kaggle":
        success = download_via_kaggle(args.data_path)
    elif args.method == "network-intrusion":
        success = download_network_intrusion_dataset(args.data_path)
    elif args.method == "wget":
        if not args.url:
            logger.error("--url required for wget method")
            sys.exit(1)
        success = download_via_wget(args.url, args.data_path)
    elif args.method == "curl":
        if not args.url:
            logger.error("--url required for curl method")
            sys.exit(1)
        success = download_via_curl(args.url, args.data_path)
    else:
        logger.info("Manual download instructions:")
        logger.info("")
        logger.info("1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
        logger.info("2. Register and download the dataset")
        logger.info("3. Extract CSV files to: ./data/raw/")
        logger.info("")
        logger.info("Alternative sources:")
        logger.info("")
        logger.info("1. CIC-IDS2017 via Kaggle:")
        logger.info("   python scripts/download_dataset.py --method kaggle")
        logger.info("")
        logger.info("2. Network Intrusion Dataset (alternative):")
        logger.info("   python scripts/download_dataset.py --method network-intrusion")
        logger.info("   Or with curl directly:")
        logger.info("   curl -L -o ~/Downloads/network-intrusion-dataset.zip \\")
        logger.info(
            "     https://www.kaggle.com/api/v1/datasets/download/"
            "chethuhn/network-intrusion-dataset"
        )
        logger.info("")
        logger.info("3. Google Drive mirrors (search for 'CIC-IDS2017 CSV')")
        logger.info("")
        success = False

    if success:
        logger.info(f"Dataset saved to {args.data_path}")
        logger.info("You can now run: python scripts/prepare_dataset.py --dataset CIC-IDS2017")
    else:
        logger.warning("Download failed. Please download manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Dataset Preparation Guide

## CIC-IDS2017 Dataset

### Automatic Download

The easiest way to download the dataset is via Kaggle:

```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (one-time setup)
# 1. Get API token from https://www.kaggle.com/account
# 2. Place kaggle.json in ~/.kaggle/
#    chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python scripts/download_dataset.py --method kaggle

# Or use the prepare script with --download flag
python scripts/prepare_dataset.py --dataset CIC-IDS2017 --download
```

### Manual Download Instructions

1. **Download the dataset**:
   - Visit: https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the CSV files (requires registration)
   - Extract to `./data/raw/` directory

2. **Expected directory structure**:
   ```
   data/
   ├── raw/
   │   ├── Monday-WorkingHours.pcap_ISCX.csv
   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   │   ├── Wednesday-WorkingHours.pcap_ISCX.csv
   │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
   │   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   └── processed/
       └── cic-ids2017_processed.csv
   ```

### Usage

```bash
# Prepare dataset (will look for CSV files in ./data/raw/)
python scripts/prepare_dataset.py --dataset CIC-IDS2017

# Specify custom raw data path
python scripts/prepare_dataset.py --dataset CIC-IDS2017 --raw-data-path /path/to/csv/files

# Load only first N files (for testing)
python scripts/prepare_dataset.py --dataset CIC-IDS2017 --max-files 2
```

### Dataset Features

The script automatically maps CIC-IDS2017 columns to our standard format:

- **Source/Destination IP**: Extracted from flow records
- **Source/Destination Port**: Port numbers
- **Protocol**: TCP (6), UDP (17), or ICMP (1)
- **Timestamp**: Flow timestamps (converted to seconds)
- **Packet Size**: Total packet length
- **Label**: BENIGN (0) or Attack (1)

### Preprocessing Steps

1. **IP Address Validation**: Removes rows with invalid or zero IPs
2. **Port Range Validation**: Clips ports to valid range (0-65535)
3. **Protocol Normalization**: Ensures valid protocol numbers
4. **Duplicate Removal**: Removes duplicate flows
5. **Timestamp Sorting**: Sorts by timestamp for temporal processing

### Placeholder Data

If no CSV files are found, the script generates placeholder data for testing. This allows you to test the training pipeline without downloading the full dataset.

### Notes

- The CIC-IDS2017 dataset is large (~2.5GB compressed)
- Processing may take several minutes depending on system resources
- The script handles multiple CSV files and combines them automatically
- Different CSV files may have slightly different column names - the script handles common variations


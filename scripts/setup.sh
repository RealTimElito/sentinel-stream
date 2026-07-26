#!/bin/bash
# Setup script for Sentinel-Stream

set -e

echo "Setting up Sentinel-Stream..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models logs certs

# Check for Redis
if ! command -v redis-server &> /dev/null; then
    echo "Warning: Redis not found. Please install Redis for the system to work."
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
fi

# Check for libpcap
if ! pkg-config --exists libpcap 2>/dev/null; then
    echo "Warning: libpcap not found. Packet capture may not work."
    echo "  Ubuntu/Debian: sudo apt-get install libpcap-dev"
    echo "  macOS: brew install libpcap"
fi

# Setup dashboard
if [ -d "dashboard" ]; then
    echo "Setting up dashboard..."
    cd dashboard
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    cd ..
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download dataset: python scripts/prepare_dataset.py --dataset CIC-IDS2017"
echo "2. Train model: python scripts/train_model.py --config configs/tgn_config.yaml"
echo "3. Start Redis: redis-server"
echo "4. Start inference API: python src/inference/api.py"
echo "5. Start dashboard: cd dashboard && npm start"
echo "6. Capture packets: sudo python scripts/capture_live.py --interface eth0"


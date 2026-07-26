#!/bin/bash
# Install system dependencies for Sentinel-Stream

set -e

echo "Installing system dependencies for Sentinel-Stream..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu system"
        echo "Installing libpcap-dev..."
        sudo apt-get update
        sudo apt-get install -y libpcap-dev
        echo "✓ libpcap-dev installed"
    elif command -v yum &> /dev/null; then
        echo "Detected RedHat/CentOS system"
        echo "Installing libpcap-devel..."
        sudo yum install -y libpcap-devel
        echo "✓ libpcap-devel installed"
    elif command -v pacman &> /dev/null; then
        echo "Detected Arch Linux system"
        echo "Installing libpcap..."
        sudo pacman -S --noconfirm libpcap
        echo "✓ libpcap installed"
    else
        echo "Unknown Linux distribution. Please install libpcap-dev manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Detected macOS system"
        echo "Installing libpcap..."
        brew install libpcap
        echo "✓ libpcap installed"
    else
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh"
        exit 1
    fi
else
    echo "Unsupported OS: $OSTYPE"
    echo "Please install libpcap development headers manually."
    exit 1
fi

echo ""
echo "System dependencies installed successfully!"
echo "You can now run: conda env create -f environment.yml"


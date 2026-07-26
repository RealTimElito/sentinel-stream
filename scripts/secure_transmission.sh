#!/bin/bash
# Secure transmission layer using socat with OpenSSL
# "The Tim Touch" - Secure data pipeline

set -e

# Configuration
CAPTURE_HOST="${CAPTURE_HOST:-localhost}"
CAPTURE_PORT="${CAPTURE_PORT:-9000}"
INFERENCE_HOST="${INFERENCE_HOST:-localhost}"
INFERENCE_PORT="${INFERENCE_PORT:-9001}"
CERT_DIR="${CERT_DIR:-./certs}"
KEY_FILE="${KEY_FILE:-$CERT_DIR/server.key}"
CERT_FILE="${CERT_FILE:-$CERT_DIR/server.crt}"

# Create certificate directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Generate self-signed certificate if it doesn't exist
if [ ! -f "$KEY_FILE" ] || [ ! -f "$CERT_FILE" ]; then
    echo "Generating self-signed certificate..."
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes -subj "/CN=sentinel-stream"
    chmod 600 "$KEY_FILE"
    chmod 644 "$CERT_FILE"
fi

echo "Starting secure transmission layer..."
echo "Capture endpoint: $CAPTURE_HOST:$CAPTURE_PORT"
echo "Inference endpoint: $INFERENCE_HOST:$INFERENCE_PORT"

# Start socat listener on inference side (receives encrypted data)
# This would typically run on the inference node
socat -d -d \
    OPENSSL-LISTEN:$INFERENCE_PORT,cert=$CERT_FILE,key=$KEY_FILE,verify=0,fork,reuseaddr \
    TCP:$CAPTURE_HOST:$CAPTURE_PORT

# On the capture side, you would run:
# socat TCP-LISTEN:$CAPTURE_PORT,fork,reuseaddr \
#     OPENSSL-CONNECT:$INFERENCE_HOST:$INFERENCE_PORT,cafile=$CERT_FILE


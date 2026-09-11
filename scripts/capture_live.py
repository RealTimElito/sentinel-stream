#!/usr/bin/env python3
"""Live packet capture script."""

import argparse
import logging
import signal
import sys

from src.ingestion.capture import PacketCapture, capture_with_scapy_fallback
from src.ingestion.redis_buffer import RedisBuffer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Capture live network packets")
    parser.add_argument(
        "--interface", type=str, required=True, help="Network interface to capture on (e.g., eth0)"
    )
    parser.add_argument("--filter", type=str, default="ip", help="BPF filter string")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--stream-name", type=str, default="network_flows", help="Redis stream name"
    )

    args = parser.parse_args()

    # Initialize Redis buffer
    redis_buffer = RedisBuffer(
        host=args.redis_host, port=args.redis_port, stream_name=args.stream_name
    )

    # Packet callback
    def handle_packet(metadata):
        """Handle captured packet."""
        try:
            msg_id = redis_buffer.add_packet(metadata)
            if msg_id:
                logger.debug(f"Added packet to Redis: {msg_id}")
        except Exception as e:
            logger.error(f"Failed to add packet: {e}")

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start capture
    logger.info(f"Starting capture on {args.interface}...")

    try:
        capture = PacketCapture(interface=args.interface, filter_str=args.filter)
        capture.capture_loop(handle_packet)
    except ImportError:
        logger.warning("pcap not available, using scapy fallback")
        capture_with_scapy_fallback(args.interface, handle_packet)
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

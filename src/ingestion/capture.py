"""Secure packet capture using libpcap."""

import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pcap
except ImportError:
    pcap = None

logger = logging.getLogger(__name__)


@dataclass
class PacketMetadata:
    """Metadata extracted from captured packet."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    packet_size: int
    tcp_flags: Optional[int] = None


class PacketCapture:
    """Secure packet capture using libpcap."""

    def __init__(self, interface: str, filter_str: Optional[str] = None, buffer_size: int = 65535):
        """
        Initialize packet capture.

        Args:
            interface: Network interface to capture on
            filter_str: BPF filter string (e.g., "tcp port 80")
            buffer_size: Capture buffer size
        """
        self.interface = interface
        self.filter_str = filter_str or "ip"
        self.buffer_size = buffer_size
        self.capture_handle = None

    def start(self):
        """Start packet capture."""
        if pcap is None:
            raise ImportError(
                "pcap module not available. Install with: "
                "pip install pypcap or use scapy fallback"
            )

        try:
            self.capture_handle = pcap.pcap(
                name=self.interface, snaplen=self.buffer_size, promisc=True, immediate=True
            )
            self.capture_handle.setfilter(self.filter_str)
            logger.info(f"Started capture on {self.interface}")
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            raise

    def stop(self):
        """Stop packet capture."""
        if self.capture_handle:
            self.capture_handle.close()
            self.capture_handle = None
            logger.info("Stopped packet capture")

    def extract_metadata(self, packet: bytes, timestamp: float) -> Optional[PacketMetadata]:
        """
        Extract metadata from raw packet.

        Args:
            packet: Raw packet bytes
            timestamp: Packet timestamp

        Returns:
            PacketMetadata or None if extraction fails
        """
        try:
            # Parse Ethernet header (14 bytes)
            if len(packet) < 14:
                return None

            eth_header = struct.unpack("!6s6sH", packet[:14])
            eth_type = eth_header[2]

            # Only process IP packets
            if eth_type != 0x0800:  # IPv4
                return None

            # Parse IP header (20 bytes minimum)
            if len(packet) < 34:
                return None

            ip_header = struct.unpack("!BBHHHBBH4s4s", packet[14:34])
            version_ihl = ip_header[0]
            protocol = ip_header[6]
            src_ip = socket.inet_ntoa(ip_header[8])
            dst_ip = socket.inet_ntoa(ip_header[9])

            # Extract IP header length
            ihl = (version_ihl & 0x0F) * 4

            # Parse TCP/UDP header
            src_port = 0
            dst_port = 0
            tcp_flags = None

            if protocol == 6:  # TCP
                if len(packet) < 14 + ihl + 20:
                    return None
                tcp_header = struct.unpack("!HHLLBBHHH", packet[14 + ihl : 14 + ihl + 20])
                src_port = tcp_header[0]
                dst_port = tcp_header[1]
                tcp_flags = tcp_header[5]
            elif protocol == 17:  # UDP
                if len(packet) < 14 + ihl + 8:
                    return None
                udp_header = struct.unpack("!HHHH", packet[14 + ihl : 14 + ihl + 8])
                src_port = udp_header[0]
                dst_port = udp_header[1]

            return PacketMetadata(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                packet_size=len(packet),
                tcp_flags=tcp_flags,
            )
        except Exception as e:
            logger.debug(f"Failed to extract metadata: {e}")
            return None

    def capture_loop(self, callback):
        """
        Main capture loop.

        Args:
            callback: Function to call with PacketMetadata for each packet
        """
        if not self.capture_handle:
            self.start()

        try:
            for timestamp, packet in self.capture_handle:
                metadata = self.extract_metadata(packet, timestamp)
                if metadata:
                    callback(metadata)
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
        finally:
            self.stop()


def capture_with_scapy_fallback(interface: str, callback):
    """
    Fallback capture using scapy if pcap is not available.

    Args:
        interface: Network interface
        callback: Function to call with PacketMetadata
    """
    try:
        from scapy.all import sniff, IP, TCP, UDP

        def process_packet(packet):
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                protocol = packet[IP].proto

                src_port = 0
                dst_port = 0
                tcp_flags = None

                if TCP in packet:
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    tcp_flags = packet[TCP].flags
                elif UDP in packet:
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport

                metadata = PacketMetadata(
                    timestamp=time.time(),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=protocol,
                    packet_size=len(packet),
                    tcp_flags=tcp_flags,
                )
                callback(metadata)

        sniff(iface=interface, prn=process_packet, store=False)
    except ImportError:
        logger.error("Neither pcap nor scapy available for packet capture")
        raise

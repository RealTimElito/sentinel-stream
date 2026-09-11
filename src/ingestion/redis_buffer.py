"""Redis buffer for streaming network data."""

import logging
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from .capture import PacketMetadata

logger = logging.getLogger(__name__)


class RedisBuffer:
    """Redis buffer for streaming packet metadata."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        stream_name: str = "network_flows",
        max_length: int = 10000,
    ):
        """
        Initialize Redis buffer.

        Args:
            host: Redis host
            port: Redis port
            stream_name: Name of the Redis stream
            max_length: Maximum length of the stream
        """
        self.host = host
        self.port = port
        self.stream_name = stream_name
        self.max_length = max_length
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.host, port=self.port, decode_responses=False, socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def add_packet(self, metadata: PacketMetadata) -> Optional[str]:
        """
        Add packet metadata to Redis stream.

        Args:
            metadata: PacketMetadata object

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.redis_client:
            self._connect()

        try:
            message = {
                b"timestamp": str(metadata.timestamp).encode(),
                b"src_ip": metadata.src_ip.encode(),
                b"dst_ip": metadata.dst_ip.encode(),
                b"src_port": str(metadata.src_port).encode(),
                b"dst_port": str(metadata.dst_port).encode(),
                b"protocol": str(metadata.protocol).encode(),
                b"packet_size": str(metadata.packet_size).encode(),
            }

            if metadata.tcp_flags is not None:
                message[b"tcp_flags"] = str(metadata.tcp_flags).encode()

            msg_id = self.redis_client.xadd(
                self.stream_name, message, maxlen=self.max_length, approximate=True
            )
            return msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        except Exception as e:
            logger.error(f"Failed to add packet to Redis: {e}")
            return None

    def read_messages(
        self, consumer_group: str, consumer_name: str, count: int = 10, block: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Read messages from Redis stream using consumer groups.

        Args:
            consumer_group: Consumer group name
            consumer_name: Consumer name
            count: Number of messages to read
            block: Block time in milliseconds

        Returns:
            List of messages
        """
        if not self.redis_client:
            self._connect()

        try:
            # Create consumer group if it doesn't exist
            try:
                self.redis_client.xgroup_create(
                    self.stream_name, consumer_group, id="0", mkstream=True
                )
            except redis.ResponseError:
                # Group already exists
                pass

            # Read messages
            messages = self.redis_client.xreadgroup(
                consumer_group, consumer_name, {self.stream_name: ">"}, count=count, block=block
            )

            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    decoded_msg = {
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                        "stream": stream.decode() if isinstance(stream, bytes) else stream,
                        "fields": {
                            k.decode() if isinstance(k, bytes) else k: (
                                v.decode() if isinstance(v, bytes) else v
                            )
                            for k, v in fields.items()
                        },
                    }
                    result.append(decoded_msg)

            return result
        except Exception as e:
            logger.error(f"Failed to read messages from Redis: {e}")
            return []

    def acknowledge(self, consumer_group: str, message_id: str):
        """
        Acknowledge message processing.

        Args:
            consumer_group: Consumer group name
            message_id: Message ID to acknowledge
        """
        if not self.redis_client:
            return

        try:
            self.redis_client.xack(self.stream_name, consumer_group, message_id)
        except Exception as e:
            logger.error(f"Failed to acknowledge message: {e}")

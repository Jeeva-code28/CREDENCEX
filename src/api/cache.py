# src/api/cache.py
# Thread-safe in-memory LRU & TTL cache for verified claims.

import re
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any

from src.utils import setup_logger

logger = setup_logger(__name__)


def normalize_claim_key(text: str) -> str:
    """
    Canonicalize a claim string to avoid duplicate pipeline executions
    for identical or trivially different inputs.

    - Lowercase
    - Strip punctuation and symbols
    - Collapse repeated whitespace
    """
    cleaned = text.lower().strip()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


class ClaimCache:
    """
    Thread-safe Least-Recently-Used (LRU) cache with Time-To-Live (TTL).
    Enables sub-10ms response times for recurring and viral rumors.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 86400):
        """
        Initialize the cache.

        Args:
            max_size: Max number of cached items before LRU eviction.
            ttl_seconds: Cache duration in seconds (default: 24 hours).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, claim: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached verification result if present and not expired.

        Args:
            claim: Raw claim text or URL.

        Returns:
            Cached result dict, or None if not found or expired.
        """
        key = normalize_claim_key(claim)

        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            now = time.time()

            # Check expiration
            if now - entry['stored_at'] > self.ttl_seconds:
                del self._cache[key]
                logger.debug(f"Cache expired for key: '{key[:40]}...'")
                return None

            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            logger.info(f"Cache HIT for key: '{key[:40]}...'")
            return entry['data']

    def set(self, claim: str, data: Dict[str, Any]) -> None:
        """
        Store verification result in cache.

        Args:
            claim: Raw claim text or URL.
            data: Verification result payload.
        """
        key = normalize_claim_key(claim)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest entry
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted oldest cache entry: '{evicted_key[:40]}...'")

            self._cache[key] = {
                'data': data,
                'stored_at': time.time()
            }
            logger.debug(f"Stored verification in cache for key: '{key[:40]}...'")

    def clear(self) -> int:
        """Clear all cached entries. Returns number of items cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cache entries.")
            return count

    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._cache)

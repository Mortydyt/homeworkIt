"""
Simple priority queue implementation for the crawler
"""

import heapq
from typing import List, Any, Optional


class PriorityQueue:
    """Thread-safe priority queue implementation"""

    def __init__(self):
        """Initialize priority queue"""
        self._queue: List[Any] = []
        self._index = 0

    def put(self, item: Any, priority: int = 0) -> None:
        """
        Add item to priority queue

        Args:
            item: Item to add
            priority: Priority (lower numbers = higher priority)
        """
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def get(self) -> Optional[Any]:
        """
        Get highest priority item

        Returns:
            Item or None if queue is empty
        """
        if not self._queue:
            return None

        priority, index, item = heapq.heappop(self._queue)
        return item

    def empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0

    def size(self) -> int:
        """Get queue size"""
        return len(self._queue)

    def peek(self) -> Optional[Any]:
        """
        Peek at highest priority item without removing

        Returns:
            Item or None if queue is empty
        """
        if not self._queue:
            return None

        return self._queue[0][2]
import heapq
from typing import Any, Dict, List, Optional, Tuple


class PriorityTaskQueue:
    def __init__(self):
        self._heap: List[Tuple[int, int, str, Any]] = []
        self._counter: int = 0

    def push(self, task_id: str, priority: int, payload: Any = None) -> None:
        # Priority contract: Higher integer = HIGHER priority (100 pops before 1)
        # In min-heap, pushing priority directly causes lowest priority to pop first
        heapq.heappush(self._heap, (-priority, self._counter, task_id, payload))
        self._counter += 1

    def pop(self) -> Optional[Dict[str, Any]]:
        if not self._heap:
            return None
        _, _, task_id, payload = heapq.heappop(self._heap)
        return {"task_id": task_id, "payload": payload}

    def __len__(self) -> int:
        return len(self._heap)

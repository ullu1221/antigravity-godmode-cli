import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 2.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_state_change = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed to allow probe request
            if now - self.last_state_change >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_state_change = time.time()

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.time()

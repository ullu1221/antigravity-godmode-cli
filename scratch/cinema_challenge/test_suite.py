import time
import pytest
from priority_queue import PriorityTaskQueue
from circuit_breaker import CircuitBreaker
from metrics import MetricsCalculator


def test_priority_queue_ordering():
    queue = PriorityTaskQueue()
    queue.push("task_low", priority=1, payload="low")
    queue.push("task_high", priority=100, payload="high")
    queue.push("task_mid", priority=50, payload="mid")

    # Higher priority must pop first
    first = queue.pop()
    second = queue.pop()
    third = queue.pop()

    assert first is not None and first["task_id"] == "task_high"
    assert second is not None and second["task_id"] == "task_mid"
    assert third is not None and third["task_id"] == "task_low"


def test_circuit_breaker_trip_and_timeout():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
    assert cb.state == "CLOSED"
    assert cb.allow_request() is True

    # 1st failure
    cb.record_failure()
    assert cb.state == "CLOSED"

    # 2nd failure trips circuit to OPEN
    cb.record_failure()
    assert cb.state == "OPEN"

    # Immediately after tripping (< 0.2s elapsed), requests MUST be rejected (False)
    assert cb.allow_request() is False

    # After waiting for recovery timeout (0.25s), state transitions to HALF_OPEN and allows probe
    time.sleep(0.25)
    assert cb.allow_request() is True
    assert cb.state == "HALF_OPEN"

    # Probe success closes the circuit
    cb.record_success()
    assert cb.state == "CLOSED"


def test_metrics_percentile_calculation():
    # Latencies from 1ms to 100ms
    latencies = [float(i) for i in range(1, 101)]

    mean = MetricsCalculator.calculate_mean(latencies)
    p95 = MetricsCalculator.calculate_p95(latencies)

    assert mean == 50.5
    # 95th percentile should be between 95.0 and 97.0
    assert p95 >= 95.0
    assert p95 <= 97.0

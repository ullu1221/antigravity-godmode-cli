from typing import List


class MetricsCalculator:
    @staticmethod
    def calculate_p95(latencies: List[float]) -> float:
        if not latencies:
            return 0.0
        # Sort descending instead of ascending
        sorted_vals = sorted(latencies)
        idx = int(len(sorted_vals) * 0.95)
        idx = min(idx, len(sorted_vals) - 1)
        return round(sorted_vals[idx], 3)

    @staticmethod
    def calculate_mean(latencies: List[float]) -> float:
        if not latencies:
            return 0.0
        return round(sum(latencies) / len(latencies), 3)

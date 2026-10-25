"""MACAD's five synthetic anomaly transformations."""

import numpy as np


class Injector:
    """Randomly inject one anomaly from the transformation set in the paper."""

    anomaly_types = ("seasonal", "trend", "global", "contextual", "shapelet")

    def __init__(self, sequence, anomaly_type=None):
        anomaly_type = anomaly_type or np.random.choice(self.anomaly_types)
        if anomaly_type not in self.anomaly_types:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        self.anomaly_type = anomaly_type
        self.injected_win = self.inject_anomaly(sequence, anomaly_type)

    @staticmethod
    def _segment(window):
        length = len(window)
        low = max(2, length // 10)
        high = max(low + 1, length // 2 + 1)
        segment_length = min(np.random.randint(low, high), length)
        start = np.random.randint(0, max(1, length - segment_length + 1))
        return start, start + segment_length

    @classmethod
    def inject_anomaly(cls, window, anomaly_type):
        window = np.asarray(window, dtype=np.float32).copy()
        if window.ndim == 1:
            window = window[:, None]

        start, end = cls._segment(window)
        scale = np.std(window, axis=0)
        scale[scale < 1e-6] = 1.0
        direction = np.random.choice((-1.0, 1.0))

        if anomaly_type == "seasonal":
            phase = np.linspace(0, 4 * np.pi, end - start, dtype=np.float32)[:, None]
            window[start:end] += direction * 2.0 * scale * np.sin(phase)
        elif anomaly_type == "trend":
            ramp = np.linspace(0, 3.0, end - start, dtype=np.float32)[:, None]
            window[start:end] += direction * ramp * scale
        elif anomaly_type == "global":
            window += direction * np.random.uniform(2.0, 4.0) * scale
        elif anomaly_type == "contextual":
            window[start:end] += direction * np.random.uniform(2.0, 4.0) * scale
        elif anomaly_type == "shapelet":
            segment = window[start:end].copy()
            window[start:end] = segment[::-1]
            if np.allclose(window[start:end], segment):
                window[start:end] += direction * scale

        return window

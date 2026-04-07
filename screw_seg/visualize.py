from __future__ import annotations

import cv2
import numpy as np

from .constants import CLASS_COLORS, CLASS_NAMES
from .structures import InstancePrediction


def overlay_instances(image: np.ndarray, instances: list[InstancePrediction], alpha: float = 0.45) -> np.ndarray:
    canvas = image.copy()
    for idx, instance in enumerate(instances):
        color = CLASS_COLORS[instance.class_id % len(CLASS_COLORS)]
        mask = instance.mask.astype(bool)
        canvas[mask] = (canvas[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = instance.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{idx + 1}:{CLASS_NAMES[instance.class_id]}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InstancePrediction:
    class_id: int
    score: float
    mask: np.ndarray
    box: tuple[int, int, int, int]
    source: str
    teacher_score: float | None = None
    seg_score: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int

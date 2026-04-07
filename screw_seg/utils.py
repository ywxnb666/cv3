from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from .constants import IMAGE_SUFFIXES


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def polygon_to_mask(polygon: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32)
    if pts.ndim != 2 or len(pts) < 3:
        return mask
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygons(mask: np.ndarray, min_points: int = 3, epsilon_ratio: float = 0.002) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < min_points:
            continue
        epsilon = max(cv2.arcLength(contour, True) * epsilon_ratio, 1.0)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
        if len(approx) >= min_points:
            polygons.append(approx)
    return polygons


def bbox_xyxy_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def flatten_polygon(points: np.ndarray, width: int, height: int) -> list[float]:
    flat: list[float] = []
    for x, y in points:
        flat.append(float(np.clip(x / width, 0.0, 1.0)))
        flat.append(float(np.clip(y / height, 0.0, 1.0)))
    return flat


def chunks(items: list, chunk_size: int) -> Iterable[list]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]

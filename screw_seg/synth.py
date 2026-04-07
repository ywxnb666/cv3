from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import CLASS_NAMES
from .utils import ensure_dir, list_images, mask_to_polygons, save_image, seed_everything, write_json


@dataclass
class Asset:
    class_id: int
    rgba: np.ndarray


def load_assets(asset_dir: Path) -> list[Asset]:
    assets: list[Asset] = []
    for class_id, _ in enumerate(CLASS_NAMES):
        class_dir = asset_dir / f"type_{class_id + 1}"
        if not class_dir.exists():
            continue
        for image_path in list_images(class_dir):
            rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            assets.append(Asset(class_id=class_id, rgba=rgba))
    if not assets:
        raise FileNotFoundError(f"No RGBA instance assets found in {asset_dir}")
    return assets


def transform_asset(asset: Asset, rng: random.Random, scale_range: tuple[float, float]) -> Asset:
    rgba = asset.rgba.copy()
    angle = rng.uniform(-180, 180)
    scale = rng.uniform(*scale_range)
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    transformed = cv2.warpAffine(
        rgba,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Asset(class_id=asset.class_id, rgba=transformed)


def composite_rgba(background: np.ndarray, rgba: np.ndarray, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgba.shape[:2]
    roi = background[y1 : y1 + h, x1 : x1 + w]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = roi.astype(np.float32) * (1.0 - alpha) + rgba[:, :, :3].astype(np.float32) * alpha
    background[y1 : y1 + h, x1 : x1 + w] = out.astype(np.uint8)
    return background, (rgba[:, :, 3] > 0).astype(np.uint8)


def sample_position(rng: random.Random, image_w: int, image_h: int, obj_w: int, obj_h: int, crowded: bool) -> tuple[int, int]:
    if crowded:
        cx = int(rng.uniform(image_w * 0.28, image_w * 0.72))
        cy = int(rng.uniform(image_h * 0.28, image_h * 0.72))
        return max(0, min(image_w - obj_w, cx - obj_w // 2)), max(0, min(image_h - obj_h, cy - obj_h // 2))
    return rng.randint(0, max(0, image_w - obj_w)), rng.randint(0, max(0, image_h - obj_h))


def generate_synthetic_dataset(
    asset_dir: Path,
    output_dir: Path,
    num_images: int = 360,
    image_size: tuple[int, int] = (1200, 1670),
    crowded_ratio: float = 0.3,
    seed: int = 3407,
) -> dict:
    seed_everything(seed)
    rng = random.Random(seed)
    assets = load_assets(asset_dir)
    images_dir = ensure_dir(output_dir / "images")
    labels_dir = ensure_dir(output_dir / "labels")
    metadata_dir = ensure_dir(output_dir / "metadata")
    width, height = image_size
    report = {"images": num_images, "instances": 0, "crowded_images": 0}
    for image_idx in range(num_images):
        crowded = image_idx < math.ceil(num_images * crowded_ratio)
        if crowded:
            report["crowded_images"] += 1
        background = np.full((height, width, 3), 255, dtype=np.uint8)
        object_count = rng.randint(10, 16) if crowded else rng.randint(14, 24)
        annotations: list[dict] = []
        metadata = {"crowded": crowded, "objects": []}
        for _ in range(object_count):
            asset = rng.choice(assets)
            transformed = transform_asset(asset, rng, scale_range=(0.7, 1.3))
            rgba = transformed.rgba
            if rgba.shape[0] < 8 or rgba.shape[1] < 8 or rgba.shape[0] >= height or rgba.shape[1] >= width:
                continue
            x1, y1 = sample_position(rng, width, height, rgba.shape[1], rgba.shape[0], crowded=crowded)
            background, local_mask = composite_rgba(background, rgba, x1, y1)
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y1 : y1 + local_mask.shape[0], x1 : x1 + local_mask.shape[1]] = local_mask
            polygons = mask_to_polygons(full_mask)
            if not polygons:
                continue
            polygon = max(polygons, key=lambda poly: cv2.contourArea(poly.astype(np.float32)))
            annotations.append({"class_id": transformed.class_id, "polygon": polygon})
            metadata["objects"].append(
                {
                    "class_id": transformed.class_id,
                    "box": [int(x1), int(y1), int(x1 + rgba.shape[1]), int(y1 + rgba.shape[0])],
                }
            )
            report["instances"] += 1
        image_path = images_dir / f"synth_{image_idx:04d}.jpg"
        save_image(image_path, background)
        from .annotation_io import save_yolo_segmentation

        save_yolo_segmentation(labels_dir / f"{image_path.stem}.txt", annotations, width=width, height=height)
        write_json(metadata_dir / f"{image_path.stem}.json", metadata)
    write_json(output_dir / "summary.json", report)
    return report

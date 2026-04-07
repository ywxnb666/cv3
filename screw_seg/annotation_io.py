from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import CLASS_NAMES
from .utils import ensure_dir, flatten_polygon, list_images, load_image, mask_to_polygons, polygon_to_mask, read_json


def load_yolo_segmentation(label_path: Path, width: int, height: int) -> list[dict]:
    rows: list[dict] = []
    if not label_path.exists():
        return rows
    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        class_id = int(float(parts[0]))
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        rows.append({"class_id": class_id, "polygon": coords})
    return rows


def _build_label_mapping() -> dict[str, int]:
    mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    for idx, _ in enumerate(CLASS_NAMES, start=1):
        mapping[f"type{idx}"] = idx - 1
        mapping[f"type_{idx}"] = idx - 1
        mapping[f"Type{idx}"] = idx - 1
    return mapping


def load_labelme_shapes(json_path: Path) -> tuple[np.ndarray, list[dict]]:
    payload = read_json(json_path)
    image_path = json_path.parent / payload["imagePath"]
    image = load_image(image_path)
    annotations: list[dict] = []
    class_to_id = _build_label_mapping()
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        label_key = label.replace(" ", "")
        if label_key.lower() == "type0":
            continue
        if label_key not in class_to_id and label_key.lower() not in class_to_id:
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        class_id = class_to_id.get(label_key, class_to_id.get(label_key.lower()))
        annotations.append({"class_id": class_id, "polygon": points})
    return image, annotations


def save_yolo_segmentation(label_path: Path, annotations: list[dict], width: int, height: int) -> None:
    lines: list[str] = []
    for ann in annotations:
        polygon = ann["polygon"]
        flat = flatten_polygon(polygon, width=width, height=height)
        if len(flat) < 6:
            continue
        row = " ".join([str(int(ann["class_id"]))] + [f"{v:.6f}" for v in flat])
        lines.append(row)
    ensure_dir(label_path.parent)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def convert_labelme_dir_to_yolo(src_dir: Path, out_dir: Path) -> dict:
    images_dir = ensure_dir(out_dir / "images")
    labels_dir = ensure_dir(out_dir / "labels")
    report = {"images": 0, "instances": 0, "classes": CLASS_NAMES}
    for json_path in sorted(src_dir.glob("*.json")):
        payload = read_json(json_path)
        image, annotations = load_labelme_shapes(json_path)
        image_path = src_dir / payload["imagePath"]
        output_image = images_dir / image_path.name
        cv2.imwrite(str(output_image), image)
        save_yolo_segmentation(labels_dir / f"{output_image.stem}.txt", annotations, width=image.shape[1], height=image.shape[0])
        report["images"] += 1
        report["instances"] += len(annotations)
    return report


def extract_instance_assets(dataset_dir: Path, output_dir: Path, margin: int = 24) -> dict:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    report = {"instances": 0, "classes": {}}
    for image_path in list_images(images_dir):
        image = load_image(image_path)
        height, width = image.shape[:2]
        labels = load_yolo_segmentation(labels_dir / f"{image_path.stem}.txt", width=width, height=height)
        for idx, ann in enumerate(labels):
            mask = polygon_to_mask(ann["polygon"], (height, width))
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)
            crop = image[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2]
            rgba = np.dstack([crop, crop_mask.astype(np.uint8) * 255])
            class_dir = ensure_dir(output_dir / f"type_{ann['class_id'] + 1}")
            out_path = class_dir / f"{image_path.stem}_{idx:04d}.png"
            cv2.imwrite(str(out_path), rgba)
            report["instances"] += 1
            key = CLASS_NAMES[ann["class_id"]]
            report["classes"][key] = report["classes"].get(key, 0) + 1
    return report


def coco_mask_to_polygon(binary_mask: np.ndarray) -> list[np.ndarray]:
    return mask_to_polygons(binary_mask.astype(np.uint8))

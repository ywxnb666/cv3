from __future__ import annotations

CLASS_NAMES = ["Type_1", "Type_2", "Type_3", "Type_4", "Type_5"]

CLASS_COLORS = [
    (230, 57, 70),
    (29, 78, 216),
    (34, 197, 94),
    (245, 158, 11),
    (139, 92, 246),
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_CONFIG = {
    "teacher_weights": "../hw2/submission/code/models/final/best.pt",
    "seg_model": "yolo11m-seg.pt",
    "imgsz": 1280,
    "tile_size": 1024,
    "tile_overlap": 0.2,
    "teacher_conf": 0.18,
    "teacher_iou": 0.65,
    "seg_conf": 0.18,
    "seg_iou": 0.6,
    "min_mask_area": 160,
    "teacher_mask_padding": 12,
    "background_white_threshold": 245,
    "max_background_ratio": 0.82,
    "seg_box_match_iou": 0.25,
    "seg_box_match_cover": 0.55,
}

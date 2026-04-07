from __future__ import annotations

import cv2
import numpy as np

from .utils import bbox_xyxy_from_mask, clip_box


def color_artifact_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 80), (15, 255, 255))
    red2 = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (45, 255, 255))
    return cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(cv2.bitwise_or(blue, green), yellow))


def white_background_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    white = cv2.inRange(image, (threshold, threshold, threshold), (255, 255, 255))
    return (white > 0).astype(np.uint8)


def metal_foreground_mask(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    blur = cv2.GaussianBlur(l_channel, (0, 0), 1.2)
    diff = cv2.absdiff(l_channel, blur)
    _, texture_mask = cv2.threshold(diff, 5, 1, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 236, 1, cv2.THRESH_BINARY_INV)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, low_sat = cv2.threshold(sat, 80, 1, cv2.THRESH_BINARY_INV)
    mask = np.logical_and(dark_mask > 0, low_sat > 0)
    mask = np.logical_or(mask, texture_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8)


def refine_box_mask(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int = 12,
    min_area: int = 160,
    max_background_ratio: float = 0.82,
    white_threshold: int = 245,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box((box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding), width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    fg = metal_foreground_mask(crop)
    artifact = color_artifact_mask(crop)
    fg[artifact > 0] = 0
    white_mask = white_background_mask(crop, threshold=white_threshold)
    if float((white_mask > 0).mean()) > max_background_ratio:
        return None
    h, w = fg.shape

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[white_mask > 0] = cv2.GC_BGD
    gc_mask[artifact > 0] = cv2.GC_BGD
    gc_mask[fg > 0] = cv2.GC_PR_FGD
    inner_margin_x = max(2, int(0.12 * w))
    inner_margin_y = max(2, int(0.12 * h))
    gc_mask[inner_margin_y : max(inner_margin_y + 1, h - inner_margin_y), inner_margin_x : max(inner_margin_x + 1, w - inner_margin_x)] = np.where(
        gc_mask[inner_margin_y : max(inner_margin_y + 1, h - inner_margin_y), inner_margin_x : max(inner_margin_x + 1, w - inner_margin_x)] == cv2.GC_BGD,
        cv2.GC_BGD,
        cv2.GC_FGD,
    )
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(crop, gc_mask, None, bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        refined = fg.astype(np.uint8)
    refined = cv2.bitwise_and(refined, cv2.bitwise_not((artifact > 0).astype(np.uint8)))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    best_mask = None
    best_score = float("-inf")
    crop_center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label_idx).astype(np.uint8)
        ys, xs = np.where(component > 0)
        center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        score = area - 0.55 * np.linalg.norm(center - crop_center)
        if score > best_score:
            best_score = score
            best_mask = component
    if best_mask is None:
        return None
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = best_mask
    final_box = bbox_xyxy_from_mask(full_mask)
    if final_box[2] <= final_box[0] or final_box[3] <= final_box[1]:
        return None
    return full_mask, final_box

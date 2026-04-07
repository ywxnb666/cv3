from __future__ import annotations

import numpy as np

from .structures import InstancePrediction
from .utils import box_iou, mask_iou


def deduplicate_instances(
    instances: list[InstancePrediction],
    mask_iou_threshold: float = 0.6,
    box_iou_threshold: float = 0.6,
) -> list[InstancePrediction]:
    ranked = sorted(
        instances,
        key=lambda row: (
            row.score,
            1.0 if row.source.startswith("seg") else 0.0,
            float(row.mask.sum()),
        ),
        reverse=True,
    )
    keep: list[InstancePrediction] = []
    for candidate in ranked:
        duplicate = False
        for existing in keep:
            if candidate.class_id != existing.class_id:
                continue
            if mask_iou(candidate.mask, existing.mask) >= mask_iou_threshold:
                duplicate = True
                break
            if box_iou(candidate.box, existing.box) >= box_iou_threshold:
                cover = np.logical_and(candidate.mask > 0, existing.mask > 0).sum()
                if cover > 0.65 * min(candidate.mask.sum(), existing.mask.sum()):
                    duplicate = True
                    break
        if not duplicate:
            keep.append(candidate)
    return keep


def attach_teacher_labels(
    instances: list[InstancePrediction],
    teacher_boxes: list[dict],
    min_box_iou: float,
    min_mask_cover: float,
) -> list[InstancePrediction]:
    assigned: list[InstancePrediction] = []
    for instance in instances:
        best = None
        best_score = -1.0
        instance_area = max(1, int(instance.mask.sum()))
        for teacher in teacher_boxes:
            overlap_iou = box_iou(instance.box, teacher["box"])
            if overlap_iou < min_box_iou:
                continue
            tx1, ty1, tx2, ty2 = teacher["box"]
            teacher_mask = np.zeros_like(instance.mask, dtype=np.uint8)
            teacher_mask[ty1:ty2, tx1:tx2] = 1
            cover = np.logical_and(instance.mask > 0, teacher_mask > 0).sum() / instance_area
            if cover < min_mask_cover:
                continue
            score = overlap_iou + 0.5 * teacher["score"] + 0.2 * cover
            if score > best_score:
                best_score = score
                best = teacher
        if best is not None:
            instance.class_id = int(best["class_id"])
            instance.teacher_score = float(best["score"])
        assigned.append(instance)
    return assigned

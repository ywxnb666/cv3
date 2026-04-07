from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .structures import InstancePrediction
from .utils import bbox_xyxy_from_mask


def _infer_config_from_checkpoint(checkpoint: Path) -> str | None:
    name = checkpoint.name.lower()
    mapping = {
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    }
    for key, value in mapping.items():
        if key in name:
            return value
    return None


@dataclass
class SamRefiner:
    checkpoint: str | None = None
    model_cfg: str | None = None
    enabled: bool = False
    available: bool = False
    device: str = "cuda"
    predictor: object | None = None
    load_error: str | None = None
    refine_sources: tuple[str, ...] = field(default_factory=lambda: ("teacher+classical", "seg-global", "seg-tile"))

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        try:
            checkpoint = Path(self.checkpoint).expanduser() if self.checkpoint else None
            if checkpoint is None or not checkpoint.exists():
                self.load_error = f"checkpoint_not_found:{self.checkpoint}"
                return
            config_name = self.model_cfg or _infer_config_from_checkpoint(checkpoint)
            if not config_name:
                self.load_error = f"config_not_inferred:{checkpoint.name}"
                return
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                config_file=config_name,
                ckpt_path=str(checkpoint),
                device=self.device,
                apply_postprocessing=True,
            )
            self.predictor = SAM2ImagePredictor(model)
            self.available = True
        except Exception as exc:
            self.load_error = repr(exc)
            self.available = False
            self.predictor = None

    def refine_instances(self, image: np.ndarray, instances: list[InstancePrediction]) -> list[InstancePrediction]:
        if not self.enabled or not self.available or self.predictor is None or not instances:
            return instances
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        refined: list[InstancePrediction] = []
        for instance in instances:
            if instance.source not in self.refine_sources:
                refined.append(instance)
                continue
            upgraded = self._refine_single(instance)
            refined.append(upgraded if upgraded is not None else instance)
        return refined

    def _refine_single(self, instance: InstancePrediction) -> InstancePrediction | None:
        if self.predictor is None:
            return None
        x1, y1, x2, y2 = instance.box
        if x2 <= x1 or y2 <= y1:
            return None
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        center = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            box=box,
            point_coords=center,
            point_labels=labels,
            multimask_output=True,
            normalize_coords=False,
        )
        if len(masks) == 0:
            return None
        best_idx = int(np.argmax(scores))
        best_mask = (masks[best_idx] > 0).astype(np.uint8)
        best_score = float(scores[best_idx])
        if int(best_mask.sum()) <= 0:
            return None
        new_box = bbox_xyxy_from_mask(best_mask)
        merged_meta = {**instance.meta, "sam_score": round(best_score, 6)}
        return InstancePrediction(
            class_id=instance.class_id,
            score=max(instance.score, best_score),
            teacher_score=instance.teacher_score,
            seg_score=instance.seg_score,
            box=new_box,
            mask=best_mask,
            source=f"{instance.source}+sam",
            meta=merged_meta,
        )

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .classical import refine_box_mask
from .constants import CLASS_NAMES, DEFAULT_CONFIG
from .fusion import attach_teacher_labels, deduplicate_instances
from .sam_refine import SamRefiner
from .structures import InstancePrediction
from .teacher import load_model, predict_boxes
from .tiling import generate_tiles
from .utils import bbox_xyxy_from_mask, ensure_dir, list_images, load_image, mask_to_polygons, write_json
from .visualize import overlay_instances


class InferencePipeline:
    def __init__(self, project_root: Path, config: dict):
        self.project_root = project_root
        self.config = {**DEFAULT_CONFIG, **config}
        teacher_weights = Path(self.config["teacher_weights"])
        if not teacher_weights.is_absolute():
            teacher_weights = (project_root / teacher_weights).resolve()
        self.teacher_model = load_model(teacher_weights)
        self.seg_model = None
        seg_weights = self.config.get("seg_weights")
        if seg_weights:
            seg_path = Path(seg_weights)
            if not seg_path.is_absolute():
                seg_path = (project_root / seg_path).resolve()
            if seg_path.exists():
                self.seg_model = load_model(seg_path)
        self.sam_refiner = SamRefiner(
            checkpoint=self.config.get("sam_checkpoint"),
            model_cfg=self.config.get("sam_model_cfg"),
            enabled=bool(self.config.get("use_sam", False)),
        )

    def run(self, data_dir: Path, output_dir: Path, device: str = "0") -> dict:
        images = list_images(data_dir)
        ensure_dir(output_dir)
        teacher_predictions = predict_boxes(
            model=self.teacher_model,
            image_paths=images,
            imgsz=int(self.config["imgsz"]),
            conf=float(self.config["teacher_conf"]),
            iou=float(self.config["teacher_iou"]),
            device=device,
            verbose=False,
        )
        summary = {"images": [], "classes": CLASS_NAMES}
        for image_path in images:
            image = load_image(image_path)
            instances = self._predict_single(image=image, teacher_boxes=teacher_predictions[image_path.stem], device=device)
            overlay = overlay_instances(image, instances)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_overlay.png"), overlay)
            payload = []
            for idx, instance in enumerate(instances):
                polygons = [
                    poly.reshape(-1).round(1).tolist()
                    for poly in mask_to_polygons(instance.mask.astype(np.uint8))
                ]
                payload.append(
                    {
                        "instance_id": idx + 1,
                        "class_id": instance.class_id,
                        "class_name": CLASS_NAMES[instance.class_id],
                        "score": instance.score,
                        "source": instance.source,
                        "box": list(instance.box),
                        "area": int(instance.mask.sum()),
                        "polygons": polygons,
                    }
                )
            write_json(output_dir / f"{image_path.stem}_instances.json", {"image": image_path.name, "instances": payload})
            summary["images"].append({"image": image_path.name, "instances": len(instances)})
        write_json(output_dir / "summary.json", summary)
        return summary

    def _predict_single(self, image: np.ndarray, teacher_boxes: list[dict], device: str) -> list[InstancePrediction]:
        instances: list[InstancePrediction] = []
        instances.extend(self._classical_from_teacher(image, teacher_boxes))
        if self.seg_model is not None:
            instances.extend(self._segment_global(image, device=device))
            instances.extend(self._segment_tiled(image, device=device))
        instances = attach_teacher_labels(
            instances,
            teacher_boxes=teacher_boxes,
            min_box_iou=float(self.config["seg_box_match_iou"]),
            min_mask_cover=float(self.config["seg_box_match_cover"]),
        )
        instances = [row for row in instances if int(row.mask.sum()) >= int(self.config["min_mask_area"])]
        instances = self.sam_refiner.refine_instances(image=image, instances=instances)
        return deduplicate_instances(instances)

    def _classical_from_teacher(self, image: np.ndarray, teacher_boxes: list[dict]) -> list[InstancePrediction]:
        output: list[InstancePrediction] = []
        for teacher in teacher_boxes:
            refined = refine_box_mask(
                image=image,
                box=teacher["box"],
                padding=int(self.config["teacher_mask_padding"]),
                min_area=int(self.config["min_mask_area"]),
                max_background_ratio=float(self.config["max_background_ratio"]),
                white_threshold=int(self.config["background_white_threshold"]),
            )
            if refined is None:
                continue
            mask, box = refined
            output.append(
                InstancePrediction(
                    class_id=int(teacher["class_id"]),
                    score=float(teacher["score"]),
                    teacher_score=float(teacher["score"]),
                    seg_score=None,
                    box=box,
                    mask=mask,
                    source="teacher+classical",
                )
            )
        return output

    def _segment_global(self, image: np.ndarray, device: str) -> list[InstancePrediction]:
        results = self.seg_model.predict(
            source=[image],
            imgsz=int(self.config["imgsz"]),
            conf=float(self.config["seg_conf"]),
            iou=float(self.config["seg_iou"]),
            device=device,
            verbose=False,
            stream=False,
        )
        return self._instances_from_seg_results(results, image_shape=image.shape[:2], source="seg-global")

    def _segment_tiled(self, image: np.ndarray, device: str) -> list[InstancePrediction]:
        height, width = image.shape[:2]
        windows = generate_tiles(
            width=width,
            height=height,
            tile_size=int(self.config["tile_size"]),
            overlap_ratio=float(self.config["tile_overlap"]),
        )
        output: list[InstancePrediction] = []
        for window in windows:
            tile = image[window.y1 : window.y2, window.x1 : window.x2]
            results = self.seg_model.predict(
                source=[tile],
                imgsz=int(self.config["imgsz"]),
                conf=float(self.config["seg_conf"]),
                iou=float(self.config["seg_iou"]),
                device=device,
                verbose=False,
                stream=False,
            )
            for instance in self._instances_from_seg_results(results, image_shape=tile.shape[:2], source="seg-tile"):
                full_mask = np.zeros((height, width), dtype=np.uint8)
                local_mask = instance.mask
                full_mask[window.y1 : window.y2, window.x1 : window.x2] = local_mask
                x1, y1, x2, y2 = instance.box
                instance.mask = full_mask
                instance.box = (x1 + window.x1, y1 + window.y1, x2 + window.x1, y2 + window.y1)
                output.append(instance)
        return output

    def _instances_from_seg_results(self, results, image_shape: tuple[int, int], source: str) -> list[InstancePrediction]:
        height, width = image_shape
        output: list[InstancePrediction] = []
        for result in results:
            if result.masks is None or result.masks.data is None:
                continue
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.zeros(len(masks), dtype=int)
            scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(masks), dtype=float)
            for mask, class_id, score in zip(masks, classes, scores):
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                binary = (mask > 0.5).astype(np.uint8)
                if int(binary.sum()) < int(self.config["min_mask_area"]):
                    continue
                box = bbox_xyxy_from_mask(binary)
                output.append(
                    InstancePrediction(
                        class_id=int(class_id),
                        score=float(score),
                        teacher_score=None,
                        seg_score=float(score),
                        mask=binary,
                        box=box,
                        source=source,
                    )
                )
        return output

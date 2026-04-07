from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


def load_model(weights_path: Path | str) -> YOLO:
    return YOLO(str(weights_path))


def predict_boxes(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    verbose: bool = False,
) -> dict[str, list[dict]]:
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=verbose,
        stream=False,
        batch=1,
    )
    output: dict[str, list[dict]] = {}
    for image_path, result in zip(image_paths, results):
        rows: list[dict] = []
        if result.boxes is not None and result.boxes.xyxy is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(xyxy, cls, confs):
                rows.append(
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "box": tuple(int(v) for v in np.round(box).tolist()),
                    }
                )
        output[image_path.stem] = rows
    return output

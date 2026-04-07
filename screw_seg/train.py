from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .utils import ensure_dir, write_json


def train_segmentation_model(
    data_yaml: Path,
    output_dir: Path,
    model_name: str = "yolo11m-seg.pt",
    imgsz: int = 1280,
    epochs: int = 250,
    patience: int = 40,
    device: str = "0",
    batch: int = 2,
    workers: int = 4,
    project_name: str = "seg_train",
) -> dict:
    ensure_dir(output_dir)
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        batch=batch,
        workers=workers,
        device=device,
        project=str(output_dir),
        name=project_name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        amp=True,
        verbose=True,
    )
    summary = {
        "best": str(Path(results.save_dir) / "weights" / "best.pt"),
        "last": str(Path(results.save_dir) / "weights" / "last.pt"),
        "save_dir": str(results.save_dir),
    }
    write_json(output_dir / f"{project_name}_summary.json", summary)
    return summary

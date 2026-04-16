import csv
import json
import math
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP, INFERRED_CLASS_ID, WORKER_DEFAULT_YOLO_DEVICE
from app.services.gpu_runtime import require_gpu_device
from app.services.modeling import compute_metrics


def ensure_ultralytics_available() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Dependencia ausente: ultralytics. Adicione 'ultralytics' ao requirements.txt do worker."
        ) from exc


def resolve_training_params(context: dict | None = None) -> dict:
    context = context or {}
    training = context.get("training") or {}
    model_cfg = context.get("model") or {}
    return {
        "model": training.get("base_model") or training.get("baseModel") or model_cfg.get("base_model") or "yolo11m-seg.pt",
        "imgsz": int(training.get("imgsz") or training.get("image_size") or training.get("imageSize") or 1280),
        "epochs": int(training.get("epochs") or 180),
        "batch": training.get("batch") if training.get("batch") is not None else -1,
        "patience": int(training.get("patience") or 45),
        "optimizer": str(training.get("optimizer") or "AdamW"),
        "lr0": float(training.get("lr0") or 0.0015),
        "lrf": float(training.get("lrf") or 0.01),
        "weight_decay": float(training.get("weight_decay") or training.get("weightDecay") or 0.0005),
        "dropout": float(training.get("dropout") or 0.0),
        "device": training.get("device") or WORKER_DEFAULT_YOLO_DEVICE,
        "workers": int(training.get("workers") or 8),
        "cache": bool(training.get("cache", True)),
        "amp": bool(training.get("amp", True)),
        "seed": int(training.get("seed") or 42),
        "conf": float(training.get("conf") or 0.25),
        "iou": float(training.get("iou") or 0.6),
        "mask_threshold": float(training.get("mask_threshold") or training.get("maskThreshold") or 0.5),
        "degrees": float(training.get("degrees") or 2.0),
        "translate": float(training.get("translate") or 0.03),
        "scale": float(training.get("scale") or 0.15),
        "shear": float(training.get("shear") or 0.0),
        "perspective": float(training.get("perspective") or 0.0),
        "fliplr": float(training.get("fliplr") or 0.5),
        "flipud": float(training.get("flipud") or 0.0),
        "mosaic": float(training.get("mosaic") or 0.35),
        "mixup": float(training.get("mixup") or 0.0),
        "copy_paste": float(training.get("copy_paste") or training.get("copyPaste") or 0.0),
        "hsv_h": float(training.get("hsv_h") or training.get("hsvH") or 0.015),
        "hsv_s": float(training.get("hsv_s") or training.get("hsvS") or 0.5),
        "hsv_v": float(training.get("hsv_v") or training.get("hsvV") or 0.25),
        "close_mosaic": int(training.get("close_mosaic") or training.get("closeMosaic") or 10),
    }


def _normalize_points(contour: np.ndarray, width: int, height: int) -> str:
    values = []
    for point in contour.reshape(-1, 2):
        x = min(max(float(point[0]) / max(width - 1, 1), 0.0), 1.0)
        y = min(max(float(point[1]) / max(height - 1, 1), 0.0), 1.0)
        values.extend([f"{x:.6f}", f"{y:.6f}"])
    return " ".join(values)


def _mask_to_yolo_segments(mask: np.ndarray, class_id: int, *, min_area: int = 24) -> list[str]:
    binary = np.ascontiguousarray((mask == class_id).astype(np.uint8) * 255)
    if binary.max() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or len(contour) < 3:
            continue
        epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            continue
        line = f"{class_id} {_normalize_points(simplified, width, height)}"
        lines.append(line)
    return lines


def export_samples_to_yolo_dataset(*, loaded_samples: list[dict], split_map: dict, output_dir: str) -> dict:
    root = Path(output_dir) / "yolo_dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    loaded_by_id = {sample["id"]: sample for sample in loaded_samples}
    for split, sample_ids in split_map.items():
        for sample_id in sample_ids:
            sample = loaded_by_id[sample_id]
            stem = sample_id
            image_path = root / "images" / split / f"{stem}.png"
            label_path = root / "labels" / split / f"{stem}.txt"
            Image.fromarray(sample["image_rgb"]).save(image_path)
            lines: list[str] = []
            for class_id in ANNOTATED_CLASS_IDS:
                lines.extend(_mask_to_yolo_segments(sample["mask"], class_id))
            label_path.write_text("\n".join(lines), encoding="utf-8")

    names = {int(class_id): CLASS_MAP[class_id]["slug"] for class_id in ANNOTATED_CLASS_IDS}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(names)}",
                f"names: {json.dumps([names[i] for i in sorted(names)], ensure_ascii=False)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"root": str(root), "data_yaml": str(data_yaml)}


def train_yolo_segmentation(*, data_yaml: str, output_dir: str, run_name: str, params: dict, progress_callback: Callable | None = None, training_run_id: str | None = None) -> dict:
    ensure_ultralytics_available()
    from ultralytics import YOLO

    device = require_gpu_device(
        params.get("device"),
        operation="Treino YOLO Segmentation",
        fallback_device=WORKER_DEFAULT_YOLO_DEVICE,
    )
    model = YOLO(params["model"])
    project_dir = Path(output_dir) / "ultralytics_runs"
    results = model.train(
        data=data_yaml,
        task="segment",
        imgsz=params["imgsz"],
        epochs=params["epochs"],
        batch=params["batch"],
        patience=params["patience"],
        optimizer=params["optimizer"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        weight_decay=params["weight_decay"],
        dropout=params["dropout"],
        degrees=params["degrees"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
        perspective=params["perspective"],
        fliplr=params["fliplr"],
        flipud=params["flipud"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        copy_paste=params["copy_paste"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        close_mosaic=params["close_mosaic"],
        cache=params["cache"],
        amp=params["amp"],
        seed=params["seed"],
        pretrained=True,
        single_cls=False,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        cos_lr=True,
        device=device,
        workers=params["workers"],
        verbose=False,
    )
    save_dir = Path(results.save_dir)
    return {
        "save_dir": str(save_dir),
        "best_model_path": str(save_dir / "weights" / "best.pt"),
        "last_model_path": str(save_dir / "weights" / "last.pt"),
        "results_csv": str(save_dir / "results.csv"),
        "results_png": str(save_dir / "results.png"),
        "args_yaml": str(save_dir / "args.yaml"),
    }


def _draw_polygon_mask(mask: np.ndarray, polygon: np.ndarray, class_id: int) -> None:
    polygon_int = np.round(polygon).astype(np.int32)
    if polygon_int.ndim != 2 or polygon_int.shape[0] < 3:
        return
    cv2.fillPoly(mask, [polygon_int.reshape(-1, 1, 2)], int(class_id))


def build_yolo_class_mask(result, *, image_shape: tuple[int, int], mask_threshold: float = 0.5) -> np.ndarray:
    height, width = image_shape
    class_mask = np.full((height, width), INFERRED_CLASS_ID, dtype=np.uint8)
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.cls is None or masks.xy is None:
        return class_mask

    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(classes), dtype=np.float32)
    polygons = masks.xy
    order = np.argsort(confs)
    for idx in order:
        class_id = int(classes[idx])
        if class_id not in ANNOTATED_CLASS_IDS:
            continue
        polygon = polygons[idx]
        if polygon is None or len(polygon) < 3:
            continue
        _draw_polygon_mask(class_mask, np.asarray(polygon, dtype=np.float32), class_id)
    return class_mask


def evaluate_yolo_model_on_samples(model_path: str, samples: list[dict], *, params: dict) -> dict:
    if not samples:
        return {
            "pixel_accuracy": None,
            "mean_iou": None,
            "per_class_iou": {meta["slug"]: None for meta in CLASS_MAP.values()},
        }

    ensure_ultralytics_available()
    from ultralytics import YOLO

    device = require_gpu_device(
        params.get("device"),
        operation="Avaliacao YOLO Segmentation",
        fallback_device=WORKER_DEFAULT_YOLO_DEVICE,
    )
    model = YOLO(model_path)
    all_true = []
    all_pred = []
    for sample in samples:
        result = model.predict(
            source=sample["image_rgb"],
            task="segment",
            imgsz=params["imgsz"],
            conf=params["conf"],
            iou=params["iou"],
            retina_masks=True,
            verbose=False,
            device=device,
        )[0]
        pred_mask = build_yolo_class_mask(
            result,
            image_shape=sample["mask"].shape,
            mask_threshold=params["mask_threshold"],
        )
        all_true.append(sample["mask"].reshape(-1))
        all_pred.append(pred_mask.reshape(-1))
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def _read_csv_rows(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            clean = {}
            for key, value in row.items():
                normalized_key = (key or "").strip()
                normalized_value = (value or "").strip()
                clean[normalized_key] = normalized_value
            rows.append(clean)
    return rows


def _maybe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _find_metric_key(row: dict, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    lowered = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    for key in row.keys():
        for candidate in candidates:
            if candidate.lower() in key.lower():
                return key
    return None


def build_training_summary(*, training_run_id: str, train_artifacts: dict, params: dict, train_metrics: dict, val_metrics: dict, test_metrics: dict, split_map: dict) -> dict:
    rows = _read_csv_rows(train_artifacts.get("results_csv", ""))
    best_epoch = None
    best_seg_map5095 = None
    initial_seg_map5095 = None
    final_seg_map5095 = None
    final_losses = {}
    seg_mAP_key = None
    seg_mAP50_key = None
    precision_key = None
    recall_key = None

    if rows:
        seg_mAP_key = _find_metric_key(rows[0], ["metrics/seg(mAP50-95(M))", "metrics/seg(mAP50-95)"])
        seg_mAP50_key = _find_metric_key(rows[0], ["metrics/seg(mAP50(M))", "metrics/seg(mAP50)"])
        precision_key = _find_metric_key(rows[0], ["metrics/precision(M)", "metrics/precision(B)", "metrics/precision"])
        recall_key = _find_metric_key(rows[0], ["metrics/recall(M)", "metrics/recall(B)", "metrics/recall"])

        scored = []
        for idx, row in enumerate(rows):
            score = _maybe_float(row.get(seg_mAP_key)) if seg_mAP_key else None
            if score is not None:
                scored.append((idx, score, row))
        if scored:
            best_epoch, best_seg_map5095, best_row = max(scored, key=lambda item: item[1])
            initial_seg_map5095 = scored[0][1]
            final_seg_map5095 = scored[-1][1]
        else:
            best_row = rows[-1]

        last_row = rows[-1]
        for key in ["train/box_loss", "train/seg_loss", "train/cls_loss", "val/box_loss", "val/seg_loss", "val/cls_loss"]:
            if key in last_row:
                final_losses[key] = _maybe_float(last_row.get(key))
    else:
        best_row = {}

    executive_summary = (
        f"Treino {training_run_id}: melhor mAP50-95 de segmentacao="
        f"{(best_seg_map5095 or 0):.4f}, IoU medio de validacao={float(val_metrics.get('mean_iou') or 0):.4f} "
        f"e estrategia de planta por exclusao mantida no pos-processamento."
    )

    markdown = f"""# Resumo do treinamento {training_run_id}

## Configuração usada
- Modelo base: `{params['model']}`
- Tarefa: `YOLO Segmentation`
- Imagem (`imgsz`): `{params['imgsz']}`
- Épocas: `{params['epochs']}`
- Batch: `{params['batch']}`
- Patience: `{params['patience']}`
- Otimizador: `{params['optimizer']}`
- Estratégia da classe **planta**: inferida por exclusão de `fundo` e `coffee`

## Evolução do treino
- Melhor época: `{best_epoch if best_epoch is not None else 'n/d'}`
- mAP50-95 (seg): `{best_seg_map5095 if best_seg_map5095 is not None else 'n/d'}`
- mAP50 (seg): `{_maybe_float(best_row.get(seg_mAP50_key)) if seg_mAP50_key else 'n/d'}`
- Precisão: `{_maybe_float(best_row.get(precision_key)) if precision_key else 'n/d'}`
- Recall: `{_maybe_float(best_row.get(recall_key)) if recall_key else 'n/d'}`
- mAP50-95 inicial: `{initial_seg_map5095 if initial_seg_map5095 is not None else 'n/d'}`
- mAP50-95 final: `{final_seg_map5095 if final_seg_map5095 is not None else 'n/d'}`

## Métricas por split (pixel-level após exclusão da planta)
- Train: acc=`{train_metrics.get('pixel_accuracy')}` | mIoU=`{train_metrics.get('mean_iou')}`
- Val: acc=`{val_metrics.get('pixel_accuracy')}` | mIoU=`{val_metrics.get('mean_iou')}`
- Test: acc=`{test_metrics.get('pixel_accuracy')}` | mIoU=`{test_metrics.get('mean_iou')}`

## Leitura rápida
{executive_summary}
"""

    return {
        "training_run_id": training_run_id,
        "params": params,
        "splits": {key: len(value) for key, value in split_map.items()},
        "best_epoch": best_epoch,
        "best_seg_map50_95": best_seg_map5095,
        "initial_seg_map50_95": initial_seg_map5095,
        "final_seg_map50_95": final_seg_map5095,
        "final_losses": final_losses,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "executive_summary": executive_summary,
        "markdown": markdown,
    }

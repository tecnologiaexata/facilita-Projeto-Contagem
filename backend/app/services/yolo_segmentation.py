import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.config import (
    ANNOTATED_CLASS_IDS,
    CLASS_MAP,
    INFERRED_CLASS_ID,
    WORKER_DEFAULT_YOLO_DEVICE,
    WORKER_DEFAULT_YOLO_MODEL,
)
from app.services.gpu_runtime import require_gpu_device
from app.services.modeling import compute_metrics


DEFAULT_YOLO_BASE_MODEL = "yolo11m-seg.pt"


def ensure_ultralytics_available() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Dependencia ausente: ultralytics. Adicione 'ultralytics' ao requirements.txt do worker."
        ) from exc


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "sim"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "nao", "não"}:
        return False
    return default


def _coerce_int(value, default: int) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _align_to_stride(value: int, stride: int = 32) -> int:
    return max(stride, int(math.ceil(max(float(value), 1.0) / stride) * stride))


def _normalize_imgsz(value):
    if value in (None, "", []):
        return None

    if isinstance(value, (list, tuple)):
        parsed = [_normalize_imgsz(item) for item in value if item not in (None, "", [])]
        parsed = [item for item in parsed if item is not None]
        if len(parsed) >= 2:
            return [int(parsed[0]), int(parsed[1])]
        if len(parsed) == 1:
            return int(parsed[0])
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"native", "original", "full", "fullres", "full-res", "real", "source"}:
            return None
        parts = [part for part in re.split(r"[^0-9]+", normalized) if part]
        if len(parts) >= 2:
            return [_align_to_stride(int(parts[0])), _align_to_stride(int(parts[1]))]
        if len(parts) == 1:
            return _align_to_stride(int(parts[0]))
        return None

    try:
        return _align_to_stride(int(float(value)))
    except Exception:
        return None


def _format_imgsz(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]}x{value[1]}"
    if value in (None, ""):
        return "nativo"
    return str(value)


def _axis_windows(size: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    size = int(size)
    tile_size = max(1, min(int(tile_size), size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    if size <= tile_size:
        return [(0, size)]

    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(size - tile_size, 0) + 1, stride))
    last_start = size - tile_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + tile_size) for start in starts]


def iter_image_tiles(image_shape: tuple[int, int], tile_size: int, overlap: int) -> list[dict]:
    height, width = image_shape[:2]
    windows = []
    for row_index, (y0, y1) in enumerate(_axis_windows(height, tile_size, overlap)):
        for col_index, (x0, x1) in enumerate(_axis_windows(width, tile_size, overlap)):
            windows.append(
                {
                    "row_index": row_index,
                    "col_index": col_index,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                }
            )
    return windows


def _should_tile_image(params: dict, image_shape: tuple[int, int]) -> bool:
    tile_size = int(params.get("tile_size") or 0)
    if not params.get("tile_enabled") or tile_size <= 0:
        return False
    height, width = image_shape[:2]
    return height > tile_size or width > tile_size


def resolve_training_runtime_params(params: dict, samples: list[dict]) -> dict:
    runtime = dict(params)
    shapes = [tuple(sample["image_rgb"].shape[:2]) for sample in samples if sample.get("image_rgb") is not None]
    if not shapes:
        return runtime

    heights = [shape[0] for shape in shapes]
    widths = [shape[1] for shape in shapes]
    runtime["source_image_summary"] = {
        "count": len(shapes),
        "min_height": int(min(heights)),
        "max_height": int(max(heights)),
        "min_width": int(min(widths)),
        "max_width": int(max(widths)),
    }

    max_long_side = max(max(heights), max(widths))
    tile_size = _align_to_stride(runtime.get("tile_size") or 1280)
    tile_overlap = max(0, _coerce_int(runtime.get("tile_overlap"), 256))
    tile_overlap = min(tile_overlap, max(tile_size - 32, 0))

    auto_tile = max_long_side > tile_size
    runtime["tile_size"] = tile_size
    runtime["tile_overlap"] = tile_overlap
    runtime["tile_enabled"] = bool(runtime.get("tile_enabled")) and auto_tile
    runtime["tile_count_estimate"] = 0
    if runtime["tile_enabled"]:
        runtime["tile_count_estimate"] = int(
            sum(len(iter_image_tiles(shape, tile_size, tile_overlap)) for shape in shapes)
        )

    if runtime.get("native_resolution") and runtime.get("imgsz") is None:
        runtime["requested_imgsz"] = "native"
        runtime["tile_enabled"] = False
        runtime["imgsz"] = min(_align_to_stride(max_long_side), 2560)
        runtime["resolved_train_imgsz"] = runtime["imgsz"]
        runtime["resolution_mode"] = "native_capped_2560"
    else:
        if runtime["tile_enabled"]:
            runtime["imgsz"] = tile_size
            runtime["resolved_train_imgsz"] = tile_size
            runtime["resolution_mode"] = "explicit_tiled"
        else:
            runtime["resolution_mode"] = "fixed_2560" if int(runtime.get("imgsz") or 0) == 2560 else "explicit"
        runtime["resolved_train_imgsz"] = runtime.get("imgsz")

    if runtime["tile_enabled"] and runtime.get("batch") == -1:
        runtime["batch"] = 2
        runtime["batch_mode"] = "fixed_for_tiles"
    else:
        runtime["batch_mode"] = "user_defined" if runtime.get("batch") != -1 else "auto"
    return runtime


def resolve_prediction_imgsz(params: dict, image_shape: tuple[int, int]) -> int | list[int]:
    if _should_tile_image(params, image_shape):
        return int(params["tile_size"])
    explicit_imgsz = params.get("imgsz")
    if explicit_imgsz is not None:
        return explicit_imgsz
    height, width = image_shape[:2]
    return [_align_to_stride(height), _align_to_stride(width)]


def resolve_yolo_model_reference(reference: str | None, *, fallback: str | None = None) -> str:
    raw_value = str(reference or fallback or "").strip().strip("\"'")
    if not raw_value:
        raw_value = DEFAULT_YOLO_BASE_MODEL

    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    try:
        candidate = Path(expanded)
    except Exception:
        return expanded

    if candidate.exists():
        return str(candidate.resolve())
    return expanded


def resolve_training_params(context: dict | None = None) -> dict:
    context = context or {}
    training = context.get("training") or {}
    model_cfg = context.get("model") or {}
    raw_imgsz = training.get("imgsz")
    if raw_imgsz in (None, ""):
        raw_imgsz = training.get("image_size") or training.get("imageSize")
    normalized_imgsz = _normalize_imgsz(raw_imgsz)
    if normalized_imgsz is None:
        normalized_imgsz = 2560
    native_resolution = _coerce_bool(
        training.get("native_resolution") or training.get("nativeResolution"),
        default=False,
    )
    tile_enabled = _coerce_bool(
        training.get("tile_enabled") or training.get("tileEnabled"),
        default=False,
    )
    return {
        "model": resolve_yolo_model_reference(
            training.get("base_model") or training.get("baseModel") or model_cfg.get("base_model"),
            fallback=WORKER_DEFAULT_YOLO_MODEL or DEFAULT_YOLO_BASE_MODEL,
        ),
        "imgsz": normalized_imgsz,
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
        "native_resolution": native_resolution,
        "rect": _coerce_bool(training.get("rect"), default=True),
        "tile_enabled": tile_enabled,
        "tile_size": _coerce_int(training.get("tile_size") or training.get("tileSize"), 1280),
        "tile_overlap": _coerce_int(training.get("tile_overlap") or training.get("tileOverlap"), 256),
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


def export_samples_to_yolo_dataset(*, loaded_samples: list[dict], split_map: dict, output_dir: str, params: dict | None = None) -> dict:
    params = params or {}
    root = Path(output_dir) / "yolo_dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    loaded_by_id = {sample["id"]: sample for sample in loaded_samples}
    exported_counts = {split: 0 for split in ("train", "val", "test")}
    for split, sample_ids in split_map.items():
        for sample_id in sample_ids:
            sample = loaded_by_id[sample_id]
            windows = (
                iter_image_tiles(sample["mask"].shape, int(params["tile_size"]), int(params["tile_overlap"]))
                if _should_tile_image(params, sample["mask"].shape)
                else [{"row_index": 0, "col_index": 0, "y0": 0, "y1": sample["mask"].shape[0], "x0": 0, "x1": sample["mask"].shape[1]}]
            )
            for window in windows:
                y0, y1, x0, x1 = window["y0"], window["y1"], window["x0"], window["x1"]
                image_tile = sample["image_rgb"][y0:y1, x0:x1]
                mask_tile = sample["mask"][y0:y1, x0:x1]
                stem = sample_id
                if len(windows) > 1:
                    stem = f"{sample_id}__r{window['row_index']:02d}_c{window['col_index']:02d}"
                image_path = root / "images" / split / f"{stem}.png"
                label_path = root / "labels" / split / f"{stem}.txt"
                Image.fromarray(image_tile).save(image_path)
                lines: list[str] = []
                for class_id in ANNOTATED_CLASS_IDS:
                    lines.extend(_mask_to_yolo_segments(mask_tile, class_id))
                label_path.write_text("\n".join(lines), encoding="utf-8")
                exported_counts[split] += 1

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
    return {"root": str(root), "data_yaml": str(data_yaml), "exported_counts": exported_counts}


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
        rect=params.get("rect", True),
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


def build_yolo_prediction_maps(result, *, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    class_mask = np.full((height, width), INFERRED_CLASS_ID, dtype=np.uint8)
    score_mask = np.zeros((height, width), dtype=np.float32)
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.cls is None or masks.xy is None:
        return class_mask, score_mask

    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(classes), dtype=np.float32)
    polygons = masks.xy
    for idx in np.argsort(confs):
        class_id = int(classes[idx])
        if class_id not in ANNOTATED_CLASS_IDS:
            continue
        polygon = polygons[idx]
        if polygon is None or len(polygon) < 3:
            continue
        polygon_int = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
        if polygon_int.ndim != 2 or polygon_int.shape[0] < 3:
            continue
        candidate_region = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(candidate_region, [polygon_int.reshape(-1, 1, 2)], 1)
        candidate_mask = candidate_region.astype(bool)
        if not np.any(candidate_mask):
            continue
        confidence = float(confs[idx])
        update_mask = candidate_mask & (confidence >= score_mask)
        class_mask[update_mask] = class_id
        score_mask[update_mask] = confidence
    return class_mask, score_mask


def build_yolo_class_mask(result, *, image_shape: tuple[int, int], mask_threshold: float = 0.5) -> np.ndarray:
    class_mask, _ = build_yolo_prediction_maps(result, image_shape=image_shape)
    return class_mask


def predict_sample_class_mask(model, image_rgb: np.ndarray, *, params: dict, device: str) -> np.ndarray:
    image_shape = image_rgb.shape[:2]
    predict_imgsz = resolve_prediction_imgsz(params, image_shape)
    if not _should_tile_image(params, image_shape):
        result = model.predict(
            source=image_rgb,
            task="segment",
            imgsz=predict_imgsz,
            conf=params["conf"],
            iou=params["iou"],
            retina_masks=True,
            verbose=False,
            device=device,
        )[0]
        return build_yolo_class_mask(
            result,
            image_shape=image_shape,
            mask_threshold=params["mask_threshold"],
        )

    full_mask = np.full(image_shape, INFERRED_CLASS_ID, dtype=np.uint8)
    full_scores = np.zeros(image_shape, dtype=np.float32)
    for window in iter_image_tiles(image_shape, int(params["tile_size"]), int(params["tile_overlap"])):
        y0, y1, x0, x1 = window["y0"], window["y1"], window["x0"], window["x1"]
        tile_rgb = image_rgb[y0:y1, x0:x1]
        tile_result = model.predict(
            source=tile_rgb,
            task="segment",
            imgsz=int(params["tile_size"]),
            conf=params["conf"],
            iou=params["iou"],
            retina_masks=True,
            verbose=False,
            device=device,
        )[0]
        tile_mask, tile_scores = build_yolo_prediction_maps(tile_result, image_shape=tile_rgb.shape[:2])
        region_scores = full_scores[y0:y1, x0:x1]
        region_mask = full_mask[y0:y1, x0:x1]
        update_mask = tile_scores > region_scores
        region_scores[update_mask] = tile_scores[update_mask]
        region_mask[update_mask] = tile_mask[update_mask]
    return full_mask


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
        pred_mask = predict_sample_class_mask(model, sample["image_rgb"], params=params, device=device)
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
- Imagem (`imgsz`): `{_format_imgsz(params['imgsz'])}`
- Modo de resoluÃ§Ã£o: `{params.get('resolution_mode') or ('native_long_side' if params.get('native_resolution') else 'explicit')}`
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
        "tile_enabled": bool(params.get("tile_enabled")),
        "tile_size": params.get("tile_size"),
        "tile_overlap": params.get("tile_overlap"),
        "tile_count_estimate": params.get("tile_count_estimate"),
        "batch_mode": params.get("batch_mode"),
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

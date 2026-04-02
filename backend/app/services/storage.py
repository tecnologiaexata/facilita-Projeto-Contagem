import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from PIL import Image

from app.config import (
    ANNOTATION_COLOR_MASKS_DIR,
    ANNOTATION_IMAGES_DIR,
    ANNOTATION_IMAGE_PREVIEWS_DIR,
    ANNOTATION_MASKS_DIR,
    ANNOTATION_METADATA_DIR,
    ANNOTATION_OVERLAYS_DIR,
    ANNOTATION_OVERLAY_PREVIEWS_DIR,
    ANNOTATION_TEXTS_DIR,
    CLASS_MAP,
    CLASS_SLUG_ALIASES,
    CVAT_DIR,
    DATASET_SPLIT_DIR,
    INFERENCES_DIR,
    REQUIRED_DIRS,
    TRAINING_DIR,
    URL_ROOT_DIR,
)


DIR_MODE = 0o777
PREVIEW_MAX_EDGE = 1600
PREVIEW_JPEG_QUALITY = 82
RESAMPLING_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(DIR_MODE)
    except PermissionError:
        pass


def ensure_storage() -> None:
    for directory in REQUIRED_DIRS:
        ensure_directory(directory)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_asset_id(prefix: str = "sample") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{timestamp}_{uuid4().hex[:12]}"


def normalize_request_id(request_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", request_id or "").strip("-").lower()
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return (cleaned or uuid4().hex[:12])[:80]


def make_stable_asset_id(prefix: str, request_id: str) -> str:
    return f"{prefix}_{normalize_request_id(request_id)}"


def slugify_name(filename: str) -> str:
    stem = Path(filename).stem or "imagem"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-")
    return cleaned or "imagem"


def write_json(path: Path, payload: dict) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_class_slug(raw_value: str) -> str:
    slug = str(raw_value).strip().lower()
    return CLASS_SLUG_ALIASES.get(slug, slug)


def _empty_class_counts() -> dict[str, int]:
    return {
        meta["slug"]: 0
        for _, meta in sorted(CLASS_MAP.items(), key=lambda item: item[0])
    }


def _normalize_class_counts(values: dict | None) -> dict[str, int]:
    normalized = _empty_class_counts()
    if not isinstance(values, dict):
        return normalized

    for raw_slug, raw_value in values.items():
        slug = _canonical_class_slug(raw_slug)
        if slug not in normalized or not isinstance(raw_value, (int, float)):
            continue
        normalized[slug] += int(raw_value)

    return normalized


def _build_current_metrics(counts: dict[str, int], total_pixels: int) -> dict:
    coffee_pixels = counts["coffee"]
    planta_pixels = counts["planta"]
    fundo_pixels = counts["fundo"]
    area_mapeada_pixels = coffee_pixels + planta_pixels
    metrics = {
        "coffee_pixels": coffee_pixels,
        "planta_pixels": planta_pixels,
        "fundo_pixels": fundo_pixels,
        "area_mapeada_pixels": area_mapeada_pixels,
        "coffee_percentual_na_imagem": round((coffee_pixels / total_pixels) * 100, 2) if total_pixels else 0.0,
        "planta_percentual_na_imagem": round((planta_pixels / total_pixels) * 100, 2) if total_pixels else 0.0,
        "fundo_percentual_na_imagem": round((fundo_pixels / total_pixels) * 100, 2) if total_pixels else 0.0,
        "area_mapeada_percentual_na_imagem": round((area_mapeada_pixels / total_pixels) * 100, 2)
        if total_pixels
        else 0.0,
        "coffee_percentual_na_area_mapeada": round((coffee_pixels / area_mapeada_pixels) * 100, 2)
        if area_mapeada_pixels
        else 0.0,
        "planta_percentual_na_area_mapeada": round((planta_pixels / area_mapeada_pixels) * 100, 2)
        if area_mapeada_pixels
        else 0.0,
    }
    metrics.update(
        {
            "cafe_pixels": metrics["coffee_pixels"],
            "cafe_percentual_na_imagem": metrics["coffee_percentual_na_imagem"],
            "cafe_percentual_na_area_mapeada": metrics["coffee_percentual_na_area_mapeada"],
        }
    )
    return metrics


def _normalize_class_list(values: list | None) -> list[str]:
    if not isinstance(values, list):
        return []

    order = {meta["slug"]: class_id for class_id, meta in CLASS_MAP.items()}
    normalized = {_canonical_class_slug(value) for value in values}
    return sorted((slug for slug in normalized if slug in order), key=lambda slug: order[slug])


def _normalize_numeric_map(values: dict | None) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}

    valid_slugs = {meta["slug"] for meta in CLASS_MAP.values()}
    normalized: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        value = _canonical_class_slug(raw_value)
        if value in valid_slugs:
            normalized[str(raw_key)] = value
    return normalized


def _normalize_pixel_stats(values: dict | None) -> dict:
    if not isinstance(values, dict):
        return values or {}

    counts = _normalize_class_counts(values.get("counts"))
    total_pixels = values.get("total_pixels")
    if not isinstance(total_pixels, int):
        total_pixels = sum(counts.values())

    percentages = {
        slug: round((count / total_pixels) * 100, 2) if total_pixels else 0.0
        for slug, count in counts.items()
    }
    return {
        **values,
        "total_pixels": total_pixels,
        "counts": counts,
        "percentages": percentages,
        "coffee_metrics": _build_current_metrics(counts, total_pixels),
    }


def normalize_annotation_payload(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["annotation_classes"] = _normalize_class_list(payload.get("annotation_classes"))
    normalized["annotation_numeric_map"] = _normalize_numeric_map(payload.get("annotation_numeric_map"))
    normalized["pixel_stats"] = _normalize_pixel_stats(payload.get("pixel_stats"))
    return normalized


def normalize_inference_payload(payload: dict) -> dict:
    normalized = dict(payload)
    counts = _normalize_class_counts(payload.get("counts"))
    total_pixels = payload.get("total_pixels")
    if not isinstance(total_pixels, int):
        total_pixels = sum(counts.values())

    percentages = {
        slug: round((count / total_pixels) * 100, 2) if total_pixels else 0.0
        for slug, count in counts.items()
    }
    normalized.update(
        {
            "total_pixels": total_pixels,
            "counts": counts,
            "percentages": percentages,
            **_build_current_metrics(counts, total_pixels),
        }
    )
    return normalized


def annotation_bundle(sample_id: str) -> dict[str, Path]:
    return {
        "image": ANNOTATION_IMAGES_DIR / f"{sample_id}.png",
        "image_preview": ANNOTATION_IMAGE_PREVIEWS_DIR / f"{sample_id}.jpg",
        "mask": ANNOTATION_MASKS_DIR / f"{sample_id}.png",
        "color_mask": ANNOTATION_COLOR_MASKS_DIR / f"{sample_id}.png",
        "overlay": ANNOTATION_OVERLAYS_DIR / f"{sample_id}.png",
        "overlay_preview": ANNOTATION_OVERLAY_PREVIEWS_DIR / f"{sample_id}.jpg",
        "metadata": ANNOTATION_METADATA_DIR / f"{sample_id}.json",
        "annotation_txt": ANNOTATION_TEXTS_DIR / f"{sample_id}.txt",
        "cvat": CVAT_DIR / f"{sample_id}.xml",
    }


def inference_bundle(run_id: str) -> dict[str, Path]:
    base_dir = INFERENCES_DIR / run_id
    return {
        "base": base_dir,
        "image": base_dir / "input.png",
        "image_preview": base_dir / "input_preview.jpg",
        "mask": base_dir / "mask.png",
        "color_mask": base_dir / "colored_mask.png",
        "overlay": base_dir / "overlay.png",
        "overlay_preview": base_dir / "overlay_preview.jpg",
        "metadata": base_dir / "result.json",
    }


def storage_url(path: Path) -> str:
    relative = path.relative_to(URL_ROOT_DIR)
    return f"/{relative.as_posix()}"


def ensure_preview_image(source_path: Path, preview_path: Path) -> Path:
    if not source_path.exists():
        return source_path
    if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
        return preview_path

    ensure_directory(preview_path.parent)
    with Image.open(source_path) as source_image:
        preview = source_image.convert("RGB")
        preview.thumbnail((PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE), RESAMPLING_LANCZOS)
        preview.save(
            preview_path,
            format="JPEG",
            quality=PREVIEW_JPEG_QUALITY,
            optimize=True,
        )
    return preview_path


def serialize_annotation_record(payload: dict) -> dict:
    payload = normalize_annotation_payload(payload)
    sample_id = payload["id"]
    bundle = annotation_bundle(sample_id)
    image_preview_path = ensure_preview_image(bundle["image"], bundle["image_preview"])
    overlay_preview_path = ensure_preview_image(bundle["overlay"], bundle["overlay_preview"])
    annotation_txt_path = bundle["annotation_txt"]
    return {
        **payload,
        "image_url": storage_url(bundle["image"]),
        "image_preview_url": storage_url(image_preview_path),
        "mask_url": storage_url(bundle["mask"]),
        "color_mask_url": storage_url(bundle["color_mask"]),
        "overlay_url": storage_url(bundle["overlay"]),
        "overlay_preview_url": storage_url(overlay_preview_path),
        "annotation_txt_url": storage_url(annotation_txt_path) if annotation_txt_path.exists() else None,
        "cvat_url": storage_url(bundle["cvat"]),
        "package_url": f"/api/annotations/{sample_id}/package",
    }


def list_annotation_metadata_paths() -> list[Path]:
    ensure_storage()
    return sorted(ANNOTATION_METADATA_DIR.glob("*.json"), reverse=True)


def count_annotation_records() -> int:
    return len(list_annotation_metadata_paths())


def load_annotation_payload(sample_id: str) -> dict | None:
    metadata_path = annotation_bundle(sample_id)["metadata"]
    if not metadata_path.exists():
        return None
    return normalize_annotation_payload(read_json(metadata_path))


def load_annotation_record(sample_id: str) -> dict | None:
    payload = load_annotation_payload(sample_id)
    if payload is None:
        return None
    return serialize_annotation_record(payload)


def list_annotation_payloads(*, offset: int = 0, limit: int | None = None) -> list[dict]:
    metadata_paths = list_annotation_metadata_paths()
    end = None if limit is None else offset + limit
    return [normalize_annotation_payload(read_json(path)) for path in metadata_paths[offset:end]]


def list_annotation_records(*, offset: int = 0, limit: int | None = None) -> list[dict]:
    ensure_storage()
    return [
        serialize_annotation_record(payload)
        for payload in list_annotation_payloads(offset=offset, limit=limit)
    ]


def list_inference_records() -> list[dict]:
    ensure_storage()
    records: list[dict] = []
    for metadata_path in sorted(INFERENCES_DIR.glob("*/result.json"), reverse=True):
        payload = normalize_inference_payload(read_json(metadata_path))
        run_id = payload["id"]
        bundle = inference_bundle(run_id)
        image_preview_path = ensure_preview_image(bundle["image"], bundle["image_preview"])
        overlay_preview_path = ensure_preview_image(bundle["overlay"], bundle["overlay_preview"])
        records.append(
            {
                **payload,
                "image_url": storage_url(bundle["image"]),
                "image_preview_url": storage_url(image_preview_path),
                "mask_url": storage_url(bundle["mask"]),
                "color_mask_url": storage_url(bundle["color_mask"]),
                "overlay_url": storage_url(bundle["overlay"]),
                "overlay_preview_url": storage_url(overlay_preview_path),
            }
        )
    return records


def latest_training_report() -> dict | None:
    report_path = TRAINING_DIR / "latest.json"
    if not report_path.exists():
        return None
    return read_json(report_path)


def clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    ensure_directory(directory)


def rebuild_dataset_split(records: Iterable[dict]) -> dict[str, list[str]]:
    records = list(records)
    clear_directory(DATASET_SPLIT_DIR)
    split_map = build_split_map(records)
    for split_name, ids in split_map.items():
        image_dir = DATASET_SPLIT_DIR / split_name / "images"
        mask_dir = DATASET_SPLIT_DIR / split_name / "masks"
        ensure_directory(image_dir)
        ensure_directory(mask_dir)
        for sample_id in ids:
            bundle = annotation_bundle(sample_id)
            shutil.copy2(bundle["image"], image_dir / bundle["image"].name)
            shutil.copy2(bundle["mask"], mask_dir / bundle["mask"].name)
    return split_map


def build_split_map(records: list[dict]) -> dict[str, list[str]]:
    ordered_ids = [record["id"] for record in sorted(records, key=lambda item: item["created_at"])]
    total = len(ordered_ids)
    if total == 0:
        return {"train": [], "val": [], "test": []}
    if total == 1:
        return {"train": ordered_ids, "val": [], "test": []}
    if total == 2:
        return {"train": [ordered_ids[0]], "val": [ordered_ids[1]], "test": []}

    train_end = max(1, round(total * 0.7))
    val_size = max(1, round(total * 0.2))
    test_size = total - train_end - val_size
    if test_size <= 0:
        test_size = 1
        train_end = max(1, train_end - 1)

    val_end = min(total - test_size, train_end + val_size)
    return {
        "train": ordered_ids[:train_end],
        "val": ordered_ids[train_end:val_end],
        "test": ordered_ids[val_end:],
    }


def class_catalog() -> list[dict]:
    return [
        {"id": class_id, **meta}
        for class_id, meta in sorted(CLASS_MAP.items(), key=lambda item: item[0])
    ]

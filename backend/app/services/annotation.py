from io import BytesIO
from pathlib import Path
import zipfile
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import CLASS_MAP
from app.services.monitoring import tracked_task
from app.services.cvat import export_cvat_for_mask
from app.services.storage import (
    annotation_bundle,
    list_annotation_payloads,
    list_annotation_records,
    make_asset_id,
    now_iso,
    read_json,
    rebuild_dataset_split,
    serialize_annotation_record,
    slugify_name,
    write_json,
)


MASK_LEVELS = np.array([0, 85, 170], dtype=np.uint8)


def read_upload_image(upload: UploadFile) -> Image.Image:
    try:
        payload = upload.file.read()
        return Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Falha ao abrir arquivo: {upload.filename}") from exc


def decode_mask(mask_image: Image.Image) -> np.ndarray:
    grayscale = np.array(mask_image.convert("L"))
    distances = np.abs(grayscale[..., None].astype(np.int16) - MASK_LEVELS.astype(np.int16))
    class_ids = distances.argmin(axis=2).astype(np.uint8)
    return class_ids


def build_color_mask(class_mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for class_id, meta in CLASS_MAP.items():
        color_mask[class_mask == class_id] = meta["color"]
    return color_mask


def build_overlay(image_rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = build_color_mask(class_mask)
    overlay = image_rgb.astype(np.float32).copy()
    for class_id in CLASS_MAP:
        region = class_mask == class_id
        if not np.any(region):
            continue
        overlay[region] = (1 - alpha) * overlay[region] + alpha * color_mask[region]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_pixel_distribution(class_mask: np.ndarray) -> dict:
    total_pixels = int(class_mask.size)
    counts = {CLASS_MAP[class_id]["slug"]: int((class_mask == class_id).sum()) for class_id in CLASS_MAP}
    percentages = {
        slug: round((count / total_pixels) * 100, 2) if total_pixels else 0.0
        for slug, count in counts.items()
    }
    return {
        "total_pixels": total_pixels,
        "counts": counts,
        "percentages": percentages,
        "coffee_metrics": compute_coffee_metrics(counts, total_pixels),
    }


def compute_coffee_metrics(counts: dict[str, int], total_pixels: int) -> dict:
    cafe_pixels = counts["folhagem"] + counts["fruto"]
    descarte_pixels = counts["fundo"]
    return {
        "cafe_pixels": cafe_pixels,
        "descarte_pixels": descarte_pixels,
        "cafe_percentual_na_imagem": round((cafe_pixels / total_pixels) * 100, 2) if total_pixels else 0.0,
        "descarte_percentual_na_imagem": round((descarte_pixels / total_pixels) * 100, 2) if total_pixels else 0.0,
        "fruto_percentual_no_cafe": round((counts["fruto"] / cafe_pixels) * 100, 2) if cafe_pixels else 0.0,
        "folhagem_percentual_no_cafe": round((counts["folhagem"] / cafe_pixels) * 100, 2) if cafe_pixels else 0.0,
    }


def persist_annotation_record(
    source_image: Image.Image,
    class_mask: np.ndarray,
    *,
    sample_id: str | None = None,
    original_filename: str | None = None,
    file_label: str | None = None,
    extra_payload: dict | None = None,
) -> dict:
    sample_id = sample_id or make_asset_id("annot")
    bundle = annotation_bundle(sample_id)
    existing_payload = read_json(bundle["metadata"]) if bundle["metadata"].exists() else {}
    resolved_filename = original_filename or existing_payload.get("original_filename") or f"{sample_id}.png"
    resolved_label = file_label or existing_payload.get("file_label") or slugify_name(resolved_filename)
    image_np = np.array(source_image)
    color_mask = build_color_mask(class_mask)
    overlay = build_overlay(image_np, class_mask)

    source_image.save(bundle["image"])
    Image.fromarray(class_mask, mode="L").save(bundle["mask"])
    Image.fromarray(color_mask).save(bundle["color_mask"])
    Image.fromarray(overlay).save(bundle["overlay"])

    dataset_stats = compute_pixel_distribution(class_mask)
    payload = {
        **existing_payload,
        **(extra_payload or {}),
        "id": sample_id,
        "storage_key": sample_id,
        "file_label": resolved_label,
        "original_filename": resolved_filename,
        "created_at": existing_payload.get("created_at", now_iso()),
        "updated_at": now_iso(),
        "width": source_image.width,
        "height": source_image.height,
        "pixel_stats": dataset_stats,
    }
    write_json(bundle["metadata"], payload)
    export_cvat_for_mask(
        sample_id=sample_id,
        original_filename=resolved_filename,
        class_mask=class_mask,
        width=source_image.width,
        height=source_image.height,
        destination=bundle["cvat"],
    )
    return serialize_annotation_record(payload)


def build_annotation_package(sample_id: str) -> tuple[BytesIO, str]:
    bundle = annotation_bundle(sample_id)
    metadata_path = bundle["metadata"]
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Anotacao nao encontrada.")

    payload = read_json(metadata_path)
    package_label = payload.get("file_label") or sample_id
    archive_root = package_label
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(bundle["image"], arcname=f"{archive_root}/{package_label}.png")
        archive.write(bundle["mask"], arcname=f"{archive_root}/{package_label}_mask.png")
        archive.write(bundle["color_mask"], arcname=f"{archive_root}/{package_label}_colored_mask.png")
        archive.write(bundle["overlay"], arcname=f"{archive_root}/{package_label}_overlay.png")
        archive.write(bundle["cvat"], arcname=f"{archive_root}/{package_label}.xml")
        archive.write(bundle["metadata"], arcname=f"{archive_root}/{package_label}.json")
    buffer.seek(0)
    return buffer, f"{package_label}.zip"


def save_annotation(
    image_file: UploadFile,
    mask_file: UploadFile,
    sample_id: str | None = None,
) -> dict:
    with tracked_task(
        kind="annotation",
        label="Salvar anotacao manual",
        metadata={"filename": image_file.filename or "imagem.png"},
    ) as task:
        task.update(phase="Lendo imagem e mascara")
        source_image = read_upload_image(image_file)
        mask_image = read_upload_image(mask_file)
        if mask_image.size != source_image.size:
            mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)

        task.update(phase="Gerando mascara")
        class_mask = decode_mask(mask_image)
        task.update(phase="Persistindo anotacao")
        record = persist_annotation_record(
            source_image,
            class_mask,
            sample_id=sample_id,
            original_filename=image_file.filename or None,
            file_label=slugify_name(image_file.filename or sample_id or "imagem"),
        )
        task.update(phase="Concluido", metadata={"sample_id": record["id"]})
        return record


def delete_annotation(sample_id: str) -> dict:
    with tracked_task(
        kind="annotation_delete",
        label="Excluir anotacao",
        metadata={"sample_id": sample_id},
    ) as task:
        bundle = annotation_bundle(sample_id)
        metadata_path = bundle["metadata"]
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Anotacao nao encontrada.")

        payload = serialize_annotation_record(read_json(metadata_path))
        task.update(phase="Removendo arquivos")
        for path in bundle.values():
            _delete_file(path)

        task.update(phase="Atualizando dataset organizado")
        rebuild_dataset_split(list_annotation_records())
        task.update(phase="Concluido")
        return payload


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def annotations_summary() -> dict:
    payloads = list_annotation_payloads()
    total_images = len(payloads)
    merged_counts = {meta["slug"]: 0 for meta in CLASS_MAP.values()}
    total_pixels = 0
    for payload in payloads:
        stats = payload["pixel_stats"]
        total_pixels += stats["total_pixels"]
        for slug, count in stats["counts"].items():
            merged_counts[slug] += count

    merged_percentages = {
        slug: round((count / total_pixels) * 100, 2) if total_pixels else 0.0
        for slug, count in merged_counts.items()
    }
    return {
        "total_annotations": total_images,
        "total_pixels": total_pixels,
        "counts": merged_counts,
        "percentages": merged_percentages,
        "coffee_metrics": compute_coffee_metrics(merged_counts, total_pixels),
    }

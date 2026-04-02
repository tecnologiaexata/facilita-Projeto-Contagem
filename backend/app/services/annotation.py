from io import BytesIO
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageDraw

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP, CLASS_SLUG_ALIASES, INFERRED_CLASS_ID
from app.services.monitoring import tracked_task
from app.services.cvat import export_cvat_for_mask
from app.services.remote_assets import fetch_remote_image, fetch_remote_text
from app.services.storage import (
    annotation_bundle,
    list_annotation_payloads,
    list_annotation_records,
    load_annotation_record,
    make_asset_id,
    make_stable_asset_id,
    now_iso,
    read_json,
    rebuild_dataset_split,
    serialize_annotation_record,
    slugify_name,
    write_json,
)


MASK_LEVELS = np.array([0, 85, 170], dtype=np.uint8)
YOLO_SEGMENTATION_HINT = (
    "Use linhas no formato YOLO de poligono ou bounding box. "
    "Exemplo: 'fundo 0.10 0.20 0.30 0.40 0.50 0.25' ou "
    "'coffee 0.52 0.41 0.08 0.10'. "
    "Para ids numericos ambiguos, adicione um cabecalho como "
    "'# class-map: 0=fundo, 1=coffee'."
)
PLANT_ANNOTATION_HINT = "A classe planta nao deve ser anotada manualmente; ela e inferida por exclusao."

CLASS_NAME_TO_ID = {
    "background": 0,
    "bg": 0,
    "descarte": 0,
    "fundo": 0,
    "cafe": 1,
    "coffee": 1,
    "coffe": 1,
    "grao": 1,
    "graos": 1,
    "planta": 2,
    "plant": 2,
    "plants": 2,
}


def create_http_error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def read_upload_image(upload: UploadFile) -> Image.Image:
    try:
        payload = upload.file.read()
        return Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Falha ao abrir arquivo: {upload.filename}") from exc


def read_upload_text(upload: UploadFile) -> str:
    try:
        payload = upload.file.read()
    except Exception as exc:  # pragma: no cover - defensive
        raise create_http_error(f"Falha ao ler arquivo de anotacao: {upload.filename}") from exc

    if not payload:
        raise create_http_error("O arquivo TXT de anotacao veio vazio.")

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = payload.decode(encoding)
        except UnicodeDecodeError:
            continue
        if content.strip():
            return content

    raise create_http_error("Nao foi possivel interpretar o TXT de anotacao.")


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


def _normalize_label_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    key = re.sub(r"[^a-zA-Z0-9]+", "_", without_accents).strip("_").lower()
    return CLASS_SLUG_ALIASES.get(key, key)


def _parse_header_numeric_map(raw_line: str) -> dict[int, int] | None:
    match = re.match(r"^#\s*class-map\s*:\s*(.+)$", raw_line.strip(), flags=re.IGNORECASE)
    if not match:
        return None

    mapping: dict[int, int] = {}
    for chunk in re.split(r"[;,]", match.group(1)):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise create_http_error(f"Cabecalho class-map invalido. {YOLO_SEGMENTATION_HINT}")
        raw_numeric, raw_label = entry.split("=", 1)
        try:
            numeric_label = int(raw_numeric.strip())
        except ValueError as exc:
            raise create_http_error(f"Cabecalho class-map invalido. {YOLO_SEGMENTATION_HINT}") from exc
        normalized_label = _normalize_label_key(raw_label)
        class_id = CLASS_NAME_TO_ID.get(normalized_label)
        if class_id is None:
            raise create_http_error(
                f"Classe '{raw_label.strip()}' nao reconhecida no cabecalho class-map."
            )
        mapping[numeric_label] = class_id

    if not mapping:
        raise create_http_error(f"Cabecalho class-map vazio. {YOLO_SEGMENTATION_HINT}")

    return mapping


def _resolve_numeric_class_map(shapes: list[dict], explicit_map: dict[int, int]) -> dict[int, int]:
    numeric_labels = sorted(
        {
            int(shape["label_token"])
            for shape in shapes
            if shape["label_token"].isdigit()
        }
    )
    if not numeric_labels:
        return {}

    if explicit_map:
        missing_labels = [label for label in numeric_labels if label not in explicit_map]
        if missing_labels:
            raise create_http_error(
                "O TXT usa ids numericos sem mapeamento completo no cabecalho class-map."
            )
        return explicit_map

    unsupported_labels = [label for label in numeric_labels if label not in {0, 1}]
    if unsupported_labels:
        raise create_http_error(
            "Foram encontrados ids numericos fora do intervalo suportado. "
            f"{YOLO_SEGMENTATION_HINT}"
        )

    return {label: label for label in numeric_labels}


def _resolve_class_id(label_token: str, numeric_map: dict[int, int], line_number: int) -> int:
    if label_token.isdigit():
        numeric_label = int(label_token)
        class_id = numeric_map.get(numeric_label)
        if class_id is None:
            raise create_http_error(
                f"Linha {line_number}: id numerico '{label_token}' nao suportado. {YOLO_SEGMENTATION_HINT}"
            )
        return class_id

    class_id = CLASS_NAME_TO_ID.get(_normalize_label_key(label_token))
    if class_id is None:
        raise create_http_error(
            f"Linha {line_number}: classe '{label_token}' nao reconhecida. "
            "Use apenas fundo ou coffee."
        )
    if class_id not in ANNOTATED_CLASS_IDS:
        raise create_http_error(
            f"Linha {line_number}: a classe '{label_token}' nao pode ser anotada. {PLANT_ANNOTATION_HINT}"
        )
    return class_id


def _parse_annotation_line(raw_line: str, line_number: int) -> dict:
    cleaned = raw_line.strip()
    if not cleaned or cleaned.startswith("#"):
        raise create_http_error(f"Linha {line_number}: nenhuma anotacao valida encontrada.")

    normalized = re.sub(r"[;,]+", " ", cleaned)
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if len(tokens) < 5:
        raise create_http_error(f"Linha {line_number}: formato invalido. {YOLO_SEGMENTATION_HINT}")

    label_token = tokens[0]
    try:
        coordinates = [float(token) for token in tokens[1:]]
    except ValueError as exc:
        raise create_http_error(
            f"Linha {line_number}: coordenadas invalidas no TXT de anotacao."
        ) from exc

    if len(coordinates) == 4:
        kind = "bbox"
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        kind = "polygon"
    else:
        raise create_http_error(f"Linha {line_number}: formato invalido. {YOLO_SEGMENTATION_HINT}")

    return {
        "label_token": label_token,
        "coordinates": coordinates,
        "kind": kind,
        "line_number": line_number,
    }


def _is_normalized_coordinates(values: list[float]) -> bool:
    return bool(values) and min(values) >= -0.0001 and max(values) <= 1.0001


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _polygon_points_to_pixels(coordinates: list[float], width: int, height: int) -> list[tuple[float, float]]:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    normalized = _is_normalized_coordinates(coordinates)
    points: list[tuple[float, float]] = []

    for index in range(0, len(coordinates), 2):
        x_value = coordinates[index]
        y_value = coordinates[index + 1]
        if normalized:
            x_value *= max_x if width > 1 else 0.0
            y_value *= max_y if height > 1 else 0.0
        points.append((_clamp(x_value, 0.0, max_x), _clamp(y_value, 0.0, max_y)))

    return points


def _bbox_to_pixels(coordinates: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    center_x, center_y, box_width, box_height = coordinates

    if _is_normalized_coordinates(coordinates):
        center_x *= max_x if width > 1 else 0.0
        center_y *= max_y if height > 1 else 0.0
        box_width *= float(max(width, 1))
        box_height *= float(max(height, 1))

    half_width = box_width / 2
    half_height = box_height / 2
    return (
        _clamp(center_x - half_width, 0.0, max_x),
        _clamp(center_y - half_height, 0.0, max_y),
        _clamp(center_x + half_width, 0.0, max_x),
        _clamp(center_y + half_height, 0.0, max_y),
    )


def build_class_mask_from_txt(annotation_text: str, width: int, height: int) -> tuple[np.ndarray, dict]:
    explicit_numeric_map: dict[int, int] = {}
    shapes: list[dict] = []

    for line_number, raw_line in enumerate(annotation_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parsed_header = _parse_header_numeric_map(stripped)
        if parsed_header is not None:
            explicit_numeric_map.update(parsed_header)
            continue
        if stripped.startswith("#"):
            continue
        shapes.append(_parse_annotation_line(stripped, line_number))

    if not shapes:
        raise create_http_error(f"Nenhuma anotacao valida foi encontrada no TXT. {YOLO_SEGMENTATION_HINT}")

    numeric_map = _resolve_numeric_class_map(shapes, explicit_numeric_map)
    mask_image = Image.new("L", (width, height), int(INFERRED_CLASS_ID))
    draw = ImageDraw.Draw(mask_image)
    used_classes: set[str] = set()
    used_formats: set[str] = set()
    resolved_shapes_by_class: dict[int, list[dict]] = {class_id: [] for class_id in ANNOTATED_CLASS_IDS}

    for shape in shapes:
        class_id = _resolve_class_id(shape["label_token"], numeric_map, shape["line_number"])
        used_classes.add(CLASS_MAP[class_id]["slug"])
        resolved_shapes_by_class.setdefault(class_id, []).append(shape)

    class_order = sorted(
        ANNOTATED_CLASS_IDS,
        key=lambda class_id: (CLASS_MAP[class_id].get("draw_order", class_id), class_id),
    )

    for class_id in class_order:
        for shape in resolved_shapes_by_class.get(class_id, []):
            if shape["kind"] == "polygon":
                draw.polygon(_polygon_points_to_pixels(shape["coordinates"], width, height), fill=int(class_id))
                used_formats.add("yolo_polygon")
                continue
            draw.rectangle(_bbox_to_pixels(shape["coordinates"], width, height), fill=int(class_id))
            used_formats.add("yolo_bbox")

    return np.array(mask_image, dtype=np.uint8), {
        "annotation_format": "mixed" if len(used_formats) > 1 else next(iter(used_formats)),
        "annotation_shape_count": len(shapes),
        "annotation_classes": sorted(used_classes),
        "annotation_numeric_map": {
            str(label): CLASS_MAP[class_id]["slug"]
            for label, class_id in sorted(numeric_map.items())
        },
    }


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


def persist_annotation_from_inputs(
    source_image: Image.Image,
    *,
    original_filename: str,
    sample_id: str | None = None,
    request_id: str | None = None,
    mask_image: Image.Image | None = None,
    annotation_text: str | None = None,
) -> dict:
    resolved_sample_id = sample_id or (
        make_stable_asset_id("annot", request_id) if request_id else make_asset_id("annot")
    )
    existing_record = load_annotation_record(resolved_sample_id) if not sample_id and request_id else None
    if existing_record is not None:
        return existing_record

    extra_payload: dict = {}
    serialized_annotation_text: str | None = None
    if annotation_text is not None:
        class_mask, annotation_meta = build_class_mask_from_txt(
            annotation_text,
            source_image.width,
            source_image.height,
        )
        serialized_annotation_text = annotation_text
        extra_payload = {
            "source_type": "external_txt",
            "annotation_source": "third_party_txt",
            **annotation_meta,
        }
    elif mask_image is not None:
        resized_mask = mask_image
        if resized_mask.size != source_image.size:
            resized_mask = resized_mask.resize(source_image.size, Image.Resampling.NEAREST)
        class_mask = decode_mask(resized_mask)
        extra_payload = {
            "source_type": "legacy_manual",
            "annotation_source": "internal_editor",
        }
    else:
        raise create_http_error("Envie uma mascara ou um TXT de anotacao para salvar o item.")

    if request_id:
        extra_payload["request_id"] = request_id

    return persist_annotation_record(
        source_image,
        class_mask,
        sample_id=resolved_sample_id,
        original_filename=original_filename,
        file_label=slugify_name(original_filename or resolved_sample_id or "imagem"),
        extra_payload=extra_payload,
        annotation_text=serialized_annotation_text,
    )


def persist_annotation_record(
    source_image: Image.Image,
    class_mask: np.ndarray,
    *,
    sample_id: str | None = None,
    original_filename: str | None = None,
    file_label: str | None = None,
    extra_payload: dict | None = None,
    annotation_text: str | None = None,
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

    if annotation_text is not None:
        bundle["annotation_txt"].write_text(annotation_text, encoding="utf-8")

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
        if bundle["annotation_txt"].exists():
            archive.write(bundle["annotation_txt"], arcname=f"{archive_root}/{package_label}.txt")
    buffer.seek(0)
    return buffer, f"{package_label}.zip"


def save_annotation(
    image_file: UploadFile,
    mask_file: UploadFile | None = None,
    annotation_file: UploadFile | None = None,
    sample_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    input_kind = "external_txt" if annotation_file is not None else "manual_mask"
    with tracked_task(
        kind="annotation",
        label="Salvar item da galeria",
        metadata={
            "filename": image_file.filename or "imagem.png",
            "request_id": request_id,
            "input_kind": input_kind,
        },
    ) as task:
        task.update(phase="Lendo imagem")
        source_image = read_upload_image(image_file)
        mask_image: Image.Image | None = None
        annotation_text: str | None = None
        if annotation_file is not None:
            task.update(phase="Lendo anotacao TXT")
            annotation_text = read_upload_text(annotation_file)
            task.update(phase="Convertendo TXT em mascara")
        elif mask_file is not None:
            task.update(phase="Lendo mascara legada")
            mask_image = read_upload_image(mask_file)
        else:
            raise create_http_error("Envie uma mascara ou um TXT de anotacao para salvar o item.")

        task.update(phase="Persistindo item")
        record = persist_annotation_from_inputs(
            source_image,
            original_filename=image_file.filename or "imagem.png",
            sample_id=sample_id,
            request_id=request_id,
            mask_image=mask_image,
            annotation_text=annotation_text,
        )
        task.update(phase="Concluido", metadata={"sample_id": record["id"]})
        return record


def save_annotation_from_urls(
    *,
    image_url: str,
    annotation_txt_url: str | None = None,
    mask_image_url: str | None = None,
    sample_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    input_kind = "external_txt_url" if annotation_txt_url is not None else "manual_mask_url"
    with tracked_task(
        kind="annotation_remote",
        label="Salvar item da galeria por URL",
        metadata={
            "image_url": image_url,
            "annotation_txt_url": annotation_txt_url,
            "mask_image_url": mask_image_url,
            "request_id": request_id,
            "input_kind": input_kind,
        },
    ) as task:
        task.update(phase="Baixando imagem remota")
        source_image, source_filename = fetch_remote_image(image_url, fallback_filename="imagem-remota.png")
        annotation_text: str | None = None
        mask_image: Image.Image | None = None

        if annotation_txt_url is not None:
            task.update(phase="Baixando anotacao TXT remota")
            annotation_text, _ = fetch_remote_text(
                annotation_txt_url,
                fallback_filename="anotacao-remota.txt",
            )
            task.update(phase="Convertendo TXT remoto em mascara")
        elif mask_image_url is not None:
            task.update(phase="Baixando mascara remota")
            mask_image, _ = fetch_remote_image(
                mask_image_url,
                fallback_filename="mascara-remota.png",
            )
        else:
            raise create_http_error("Envie annotation_txt_url ou mask_image_url para salvar por URL.")

        task.update(phase="Persistindo item")
        record = persist_annotation_from_inputs(
            source_image,
            original_filename=source_filename,
            sample_id=sample_id,
            request_id=request_id,
            mask_image=mask_image,
            annotation_text=annotation_text,
        )
        task.update(phase="Concluido", metadata={"sample_id": record["id"]})
        return record


def delete_annotation(sample_id: str) -> dict:
    with tracked_task(
        kind="annotation_delete",
        label="Excluir item da galeria",
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
    source_breakdown: dict[str, int] = {}
    for payload in payloads:
        stats = payload["pixel_stats"]
        total_pixels += stats["total_pixels"]
        source_key = payload.get("source_type") or "legacy_manual"
        source_breakdown[source_key] = source_breakdown.get(source_key, 0) + 1
        for slug, count in stats["counts"].items():
            merged_counts[slug] += count

    merged_percentages = {
        slug: round((count / total_pixels) * 100, 2) if total_pixels else 0.0
        for slug, count in merged_counts.items()
    }
    return {
        "total_annotations": total_images,
        "total_gallery_items": total_images,
        "total_pixels": total_pixels,
        "counts": merged_counts,
        "percentages": merged_percentages,
        "source_breakdown": source_breakdown,
        "coffee_metrics": compute_coffee_metrics(merged_counts, total_pixels),
    }

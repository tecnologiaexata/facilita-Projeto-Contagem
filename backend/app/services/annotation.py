import re
import unicodedata

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP, CLASS_SLUG_ALIASES, INFERRED_CLASS_ID


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
GRAYSCALE_CLASS_CANDIDATES = {
    0: (0,),
    1: (1, 85, 127),
    2: (2, 170, 255),
}


def create_http_error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def decode_mask(mask_image: Image.Image) -> np.ndarray:
    image_array = np.asarray(mask_image)

    if image_array.ndim == 3:
        rgb = image_array[..., :3].astype(np.int16)
        class_ids = np.array(sorted(CLASS_MAP), dtype=np.uint8)
        palette = np.array([CLASS_MAP[class_id]["color"] for class_id in class_ids], dtype=np.int16)
        distances = np.square(rgb[..., None, :] - palette[None, None, :, :]).sum(axis=3)
        return class_ids[distances.argmin(axis=2)]

    grayscale = np.asarray(mask_image.convert("L"))
    max_class_id = max(CLASS_MAP)
    if grayscale.size and int(grayscale.max()) <= max_class_id:
        return grayscale.astype(np.uint8)

    candidates = []
    candidate_class_ids = []
    for class_id in sorted(CLASS_MAP):
        for value in GRAYSCALE_CLASS_CANDIDATES.get(class_id, (class_id,)):
            candidates.append(value)
            candidate_class_ids.append(class_id)

    distance = np.abs(grayscale[..., None].astype(np.int16) - np.array(candidates, dtype=np.int16))
    return np.array(candidate_class_ids, dtype=np.uint8)[distance.argmin(axis=2)]


def build_color_mask(class_mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for class_id, meta in CLASS_MAP.items():
        color_mask[class_mask == class_id] = meta["color"]
    return color_mask


def build_overlay(image_rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = build_color_mask(class_mask)
    overlay = image_rgb.astype(np.float32).copy()
    for class_id in sorted(CLASS_MAP):
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


def _parse_annotation_line(raw_line: str, line_number: int) -> dict:
    normalized = re.sub(r"[;,]+", " ", raw_line.strip())
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if len(tokens) < 5:
        raise create_http_error(f"Linha {line_number}: formato invalido. {YOLO_SEGMENTATION_HINT}")

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
        "label_token": tokens[0],
        "coordinates": coordinates,
        "kind": kind,
        "line_number": line_number,
    }


def _resolve_numeric_class_map(shapes: list[dict], explicit_map: dict[int, int]) -> dict[int, int]:
    numeric_labels = sorted(
        int(shape["label_token"])
        for shape in shapes
        if str(shape["label_token"]).isdigit()
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
            f"Linha {line_number}: classe '{label_token}' nao reconhecida. Use apenas fundo ou coffee."
        )
    if class_id not in ANNOTATED_CLASS_IDS:
        raise create_http_error(
            f"Linha {line_number}: a classe '{label_token}' nao pode ser anotada. {PLANT_ANNOTATION_HINT}"
        )
    return class_id


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
            else:
                draw.rectangle(_bbox_to_pixels(shape["coordinates"], width, height), fill=int(class_id))
                used_formats.add("yolo_bbox")

    annotation_format = None
    if len(used_formats) == 1:
        annotation_format = next(iter(used_formats))
    elif len(used_formats) > 1:
        annotation_format = "mixed"

    return np.asarray(mask_image, dtype=np.uint8), {
        "annotation_format": annotation_format,
        "annotation_shape_count": len(shapes),
        "annotation_classes": sorted(used_classes),
        "annotation_numeric_map": {
            str(label): CLASS_MAP[class_id]["slug"]
            for label, class_id in sorted(numeric_map.items())
        },
    }


def compute_pixel_distribution(class_mask: np.ndarray) -> dict:
    total_pixels = int(class_mask.size)
    counts = {CLASS_MAP[class_id]["slug"]: int((class_mask == class_id).sum()) for class_id in sorted(CLASS_MAP)}
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
    coffee_pixels = int(counts.get("coffee", 0))
    planta_pixels = int(counts.get("planta", 0))
    fundo_pixels = int(counts.get("fundo", 0))
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
    metrics["cafe_pixels"] = metrics["coffee_pixels"]
    metrics["cafe_percentual_na_imagem"] = metrics["coffee_percentual_na_imagem"]
    metrics["cafe_percentual_na_area_mapeada"] = metrics["coffee_percentual_na_area_mapeada"]
    return metrics

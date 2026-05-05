import base64
from io import BytesIO
import re
import unicodedata

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.config import (
    CLASS_MAP,
    CLASS_SLUG_ALIASES,
    INFERRED_CLASS_ID,
    ROBOFLOW_API_KEY,
    ROBOFLOW_API_URL,
    ROBOFLOW_CLASSES,
    ROBOFLOW_CLASSES_PARAMETER,
    ROBOFLOW_CONFIDENCE,
    ROBOFLOW_CONFIDENCE_PARAMETER,
    ROBOFLOW_IMAGE_INPUT,
    ROBOFLOW_MAX_IMAGE_SIDE,
    ROBOFLOW_TIMEOUT_SECONDS,
    ROBOFLOW_USE_CACHE,
    ROBOFLOW_WORKFLOW,
    ROBOFLOW_WORKSPACE,
)


def normalize_class_key(value) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    key = re.sub(r"[^a-zA-Z0-9]+", "_", without_accents).strip("_").lower()
    return CLASS_SLUG_ALIASES.get(key, key)


def class_id_from_label(value) -> int | None:
    if value is None:
        return None
    try:
        class_id = int(value)
    except (TypeError, ValueError):
        class_id = None
    if class_id in CLASS_MAP:
        return class_id

    normalized = normalize_class_key(value)
    for local_id, meta in CLASS_MAP.items():
        if normalized in {normalize_class_key(meta["slug"]), normalize_class_key(meta["label"])}:
            return local_id
    return None


def flatten_outputs(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from flatten_outputs(child)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_outputs(item)


def first_output(result):
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        outputs = result.get("outputs")
        if isinstance(outputs, list) and outputs:
            return outputs[0]
    return result


def decode_base64_image(value) -> Image.Image | None:
    if isinstance(value, dict):
        value = value.get("value") or value.get("data") or value.get("base64")
    if not isinstance(value, str) or not value.strip():
        return None

    encoded = value.strip()
    if "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return Image.open(BytesIO(base64.b64decode(encoded, validate=False)))
    except Exception:
        return None


def default_numeric_class_map() -> dict[int, int]:
    mapping = {}
    for index, class_name in enumerate(ROBOFLOW_CLASSES):
        class_id = class_id_from_label(class_name)
        if class_id is not None:
            mapping[index] = class_id
    return mapping


def parse_class_map(raw_class_map) -> dict[int, int]:
    if isinstance(raw_class_map, list):
        mapping = {index: class_id_from_label(value) for index, value in enumerate(raw_class_map)}
        return {key: value for key, value in mapping.items() if value is not None} or default_numeric_class_map()
    if not isinstance(raw_class_map, dict):
        return default_numeric_class_map()

    mapping = {}
    for raw_key, raw_value in raw_class_map.items():
        try:
            source_value = int(raw_key)
            target_label = raw_value
        except (TypeError, ValueError):
            try:
                source_value = int(raw_value)
            except (TypeError, ValueError):
                continue
            target_label = raw_key
        class_id = class_id_from_label(target_label)
        if class_id is not None:
            mapping[source_value] = class_id
    return mapping or default_numeric_class_map()


def semantic_mask_from_result(result, image_shape: tuple[int, int]) -> np.ndarray | None:
    height, width = image_shape
    for output in flatten_outputs(first_output(result)):
        raw_mask = output.get("segmentation_mask") or output.get("semantic_mask") or output.get("class_mask")
        if raw_mask is None:
            continue
        mask_image = decode_base64_image(raw_mask)
        if mask_image is None:
            continue
        if mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        source_mask = np.asarray(mask_image.convert("L"))
        class_mask = np.full((height, width), INFERRED_CLASS_ID, dtype=np.uint8)
        raw_class_map = output.get("class_map") or output.get("classMap") or output.get("classes")
        for source_value, target_class_id in parse_class_map(raw_class_map).items():
            class_mask[source_mask == int(source_value)] = int(target_class_id)
        return class_mask
    return None


def prediction_confidence(prediction: dict) -> float:
    for key in ("confidence", "score", "class_confidence"):
        try:
            return float(prediction.get(key))
        except (TypeError, ValueError):
            continue
    return 1.0


def prediction_class_id(prediction: dict) -> int | None:
    for key in ("class", "class_name", "label", "name", "class_id", "classId"):
        class_id = class_id_from_label(prediction.get(key))
        if class_id is not None:
            return class_id
    return None


def point_xy(point) -> tuple[float, float] | None:
    if isinstance(point, dict):
        try:
            return float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            return float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
    return None


def polygon_from_prediction(prediction: dict) -> np.ndarray | None:
    raw_points = prediction.get("points") or prediction.get("polygon") or prediction.get("vertices")
    if not isinstance(raw_points, list):
        return None
    points = [xy for xy in (point_xy(point) for point in raw_points) if xy is not None]
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float32)


def rectangle_from_prediction(prediction: dict) -> np.ndarray | None:
    try:
        center_x = float(prediction["x"])
        center_y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (KeyError, TypeError, ValueError):
        return None
    x0 = center_x - width / 2
    x1 = center_x + width / 2
    y0 = center_y - height / 2
    y1 = center_y + height / 2
    return np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float32)


def find_predictions(result) -> list[dict]:
    for output in flatten_outputs(first_output(result)):
        predictions = output.get("predictions")
        if isinstance(predictions, dict):
            predictions = predictions.get("predictions") or predictions.get("detections")
        if isinstance(predictions, list):
            return [prediction for prediction in predictions if isinstance(prediction, dict)]
    return []


def class_mask_from_predictions(predictions: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    class_mask = np.full((height, width), INFERRED_CLASS_ID, dtype=np.uint8)
    score_mask = np.zeros((height, width), dtype=np.float32)
    for prediction in sorted(predictions, key=prediction_confidence):
        class_id = prediction_class_id(prediction)
        if class_id is None:
            continue
        polygon = polygon_from_prediction(prediction)
        if polygon is None:
            polygon = rectangle_from_prediction(prediction)
        if polygon is None:
            continue
        candidate_region = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(candidate_region, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 1)
        candidate_mask = candidate_region.astype(bool)
        confidence = prediction_confidence(prediction)
        update_mask = candidate_mask & (confidence >= score_mask)
        class_mask[update_mask] = int(class_id)
        score_mask[update_mask] = confidence
    return class_mask


def class_mask_from_result(
    result,
    image_shape: tuple[int, int],
    min_confidence: float | None = None,
) -> tuple[np.ndarray, dict]:
    class_mask = semantic_mask_from_result(result, image_shape)
    if class_mask is not None:
        return class_mask, {"output_mode": "semantic_mask"}

    predictions = find_predictions(result)
    if not predictions:
        raise HTTPException(
            status_code=502,
            detail="Roboflow nao retornou segmentation_mask/class_map nem predictions com points.",
        )
    total_predictions = len(predictions)
    if min_confidence is not None:
        predictions = [
            prediction
            for prediction in predictions
            if prediction_confidence(prediction) >= min_confidence
        ]
    return class_mask_from_predictions(predictions, image_shape), {
        "output_mode": "predictions",
        "predictions_count": len(predictions),
        "raw_predictions_count": total_predictions,
        "min_confidence": min_confidence,
    }


def resize_for_roboflow(image: Image.Image) -> tuple[Image.Image, dict]:
    original_size = {"original_width": image.width, "original_height": image.height}
    max_side = int(ROBOFLOW_MAX_IMAGE_SIDE or 0)
    if not max_side or max(image.size) <= max_side:
        return image, {
            "resized": False,
            **original_size,
            "inference_width": image.width,
            "inference_height": image.height,
        }

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized, {
        "resized": True,
        **original_size,
        "inference_width": resized.width,
        "inference_height": resized.height,
        "max_image_side": max_side,
    }


def resolve_confidence(value) -> float | None:
    if value is None or value == "":
        return ROBOFLOW_CONFIDENCE
    try:
        return float(value)
    except (TypeError, ValueError):
        return ROBOFLOW_CONFIDENCE


def require_roboflow_config() -> None:
    missing = []
    if not ROBOFLOW_API_KEY:
        missing.append("ROBOFLOW_API_KEY")
    if not ROBOFLOW_WORKSPACE:
        missing.append("ROBOFLOW_WORKSPACE")
    if not ROBOFLOW_WORKFLOW:
        missing.append("ROBOFLOW_WORKFLOW")
    if missing:
        raise HTTPException(status_code=500, detail=f"Configure no worker: {', '.join(missing)}")


def encode_image_base64_jpeg(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_roboflow_workflow(image: Image.Image, parameters: dict) -> dict | list:
    try:
        import requests
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Dependencia ausente: requests. Instale as dependencias atualizadas do worker.",
        ) from exc

    url = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW}"
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "use_cache": ROBOFLOW_USE_CACHE,
        "inputs": {
            ROBOFLOW_IMAGE_INPUT: {
                "type": "base64",
                "value": encode_image_base64_jpeg(image),
            },
            **parameters,
        },
    }
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=ROBOFLOW_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Falha de rede ao chamar Roboflow: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:800] if response.text else response.reason
        raise HTTPException(status_code=502, detail=f"Roboflow retornou HTTP {response.status_code}: {detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Roboflow retornou uma resposta que nao e JSON.") from exc


def run_roboflow_inference(source_image: Image.Image, *, confidence=None) -> dict:
    require_roboflow_config()
    inference_image, image_preprocessing = resize_for_roboflow(source_image)
    image_rgb = np.asarray(inference_image)
    confidence_value = resolve_confidence(confidence)
    parameters = {}
    if ROBOFLOW_CLASSES_PARAMETER:
        parameters[ROBOFLOW_CLASSES_PARAMETER] = list(ROBOFLOW_CLASSES)
    if ROBOFLOW_CONFIDENCE_PARAMETER and confidence_value is not None:
        parameters[ROBOFLOW_CONFIDENCE_PARAMETER] = float(confidence_value)

    result = call_roboflow_workflow(inference_image, parameters)
    class_mask, output_metadata = class_mask_from_result(
        result,
        image_rgb.shape[:2],
        min_confidence=confidence_value,
    )
    return {
        "image": inference_image,
        "image_rgb": image_rgb,
        "class_mask": class_mask,
        "raw_result": result,
        "metadata": {
            "api_url": ROBOFLOW_API_URL,
            "workspace": ROBOFLOW_WORKSPACE,
            "workflow": ROBOFLOW_WORKFLOW,
            "image_input": ROBOFLOW_IMAGE_INPUT,
            "parameters": parameters,
            "image_preprocessing": image_preprocessing,
            **output_metadata,
        },
    }

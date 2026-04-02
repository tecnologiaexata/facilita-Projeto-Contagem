import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

from app.config import CLASS_MAP
from app.services.annotation import compute_coffee_metrics, compute_pixel_distribution


MAX_PIXELS_PER_CLASS = 3500
RNG = np.random.default_rng(42)


def build_features(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0
    hsv[..., 1:] /= 255.0
    height, width = image_rgb.shape[:2]
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    if width > 1:
        x_coords /= width - 1
    if height > 1:
        y_coords /= height - 1
    features = np.dstack([rgb, hsv, x_coords[..., None], y_coords[..., None]])
    return features.reshape(-1, features.shape[-1])


def sample_training_pixels(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = build_features(image_rgb)
    labels = mask.reshape(-1)
    sampled_features = []
    sampled_labels = []
    for class_id in CLASS_MAP:
        indices = np.flatnonzero(labels == class_id)
        if len(indices) == 0:
            continue
        if len(indices) > MAX_PIXELS_PER_CLASS:
            indices = RNG.choice(indices, size=MAX_PIXELS_PER_CLASS, replace=False)
        sampled_features.append(features[indices])
        sampled_labels.append(labels[indices])
    if not sampled_features:
        raise ValueError("Nao ha pixels suficientes para treino.")
    return np.vstack(sampled_features), np.concatenate(sampled_labels)


def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    label_ids = sorted(CLASS_MAP)
    matrix = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class_iou = {}
    iou_values = []
    for index, class_id in enumerate(label_ids):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        denominator = tp + fp + fn
        iou = float(tp / denominator) if denominator else 0.0
        per_class_iou[CLASS_MAP[class_id]["slug"]] = round(iou, 4)
        iou_values.append(iou)
    return {
        "pixel_accuracy": round(accuracy, 4),
        "mean_iou": round(float(np.mean(iou_values)) if iou_values else 0.0, 4),
        "per_class_iou": per_class_iou,
    }


def calculate_inference_payload(class_mask: np.ndarray) -> dict:
    distribution = compute_pixel_distribution(class_mask)
    counts = distribution["counts"]
    total = distribution["total_pixels"]
    coffee_metrics = compute_coffee_metrics(counts, total)
    return {
        **distribution,
        **coffee_metrics,
    }

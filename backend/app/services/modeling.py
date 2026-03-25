import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import cv2
import joblib
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from app.config import CLASS_MAP, MODELS_DIR, TRAINING_DIR
from app.services.annotation import (
    build_color_mask,
    build_overlay,
    compute_coffee_metrics,
    compute_pixel_distribution,
)
from app.services.monitoring import list_active_tasks, list_recent_tasks, tracked_task
from app.services.storage import (
    annotation_bundle,
    class_catalog,
    ensure_directory,
    ensure_preview_image,
    inference_bundle,
    latest_training_report,
    list_annotation_records,
    list_inference_records,
    make_asset_id,
    now_iso,
    read_json,
    rebuild_dataset_split,
    storage_url,
    write_json,
)


MODEL_PATH = MODELS_DIR / "latest.joblib"
MAX_PIXELS_PER_CLASS = 3500
RNG = np.random.default_rng(42)
ACTIVE_TRAINING_STATUSES = {"queued", "running"}


@dataclass
class LoadedModel:
    classifier: RandomForestClassifier
    trained_at: str


@dataclass
class TrainingJobState:
    id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


TRAINING_JOB_LOCK = Lock()
TRAINING_JOB_STATE: TrainingJobState | None = None


def _active_training_task() -> dict | None:
    active_tasks = list_active_tasks(kind="training")
    return active_tasks[0] if active_tasks else None


def _recent_training_task() -> dict | None:
    recent_tasks = list_recent_tasks(kind="training")
    return recent_tasks[0] if recent_tasks else None


def _default_training_phase(status: str) -> str:
    if status == "queued":
        return "Aguardando disponibilidade do backend"
    if status == "running":
        return "Treinando modelo"
    if status == "completed":
        return "Concluido"
    if status == "failed":
        return "Falha no treino"
    return "Aguardando"


def _serialize_training_job(job: TrainingJobState | None) -> dict | None:
    if job is None:
        return None

    active_task = _active_training_task()
    recent_task = (
        _recent_training_task()
        if not active_task and job.status not in ACTIVE_TRAINING_STATUSES
        else None
    )
    task_payload = active_task or recent_task
    status = "running" if active_task and job.status in ACTIVE_TRAINING_STATUSES else job.status
    phase = task_payload["phase"] if task_payload else _default_training_phase(status)

    return {
        "id": job.id,
        "status": status,
        "is_active": status in ACTIVE_TRAINING_STATUSES,
        "phase": phase,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "elapsed_seconds": task_payload.get("elapsed_seconds") if task_payload else None,
        "error": job.error or (task_payload.get("error") if task_payload else None),
        "task": task_payload,
    }


def _update_training_job(job_id: str, **changes) -> None:
    global TRAINING_JOB_STATE

    with TRAINING_JOB_LOCK:
        if TRAINING_JOB_STATE is None or TRAINING_JOB_STATE.id != job_id:
            return
        for field_name, value in changes.items():
            setattr(TRAINING_JOB_STATE, field_name, value)


def _run_training_job(job_id: str) -> None:
    _update_training_job(job_id, status="running", started_at=now_iso(), finished_at=None, error=None)
    try:
        train_latest_model()
    except HTTPException as exc:
        error_message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        _update_training_job(job_id, status="failed", finished_at=now_iso(), error=error_message)
    except Exception as exc:  # pragma: no cover - defensive
        _update_training_job(job_id, status="failed", finished_at=now_iso(), error=str(exc))
    else:
        _update_training_job(job_id, status="completed", finished_at=now_iso(), error=None)


def start_training_job() -> tuple[dict | None, bool]:
    global TRAINING_JOB_STATE

    with TRAINING_JOB_LOCK:
        current_job = TRAINING_JOB_STATE
        if current_job is not None and current_job.status in ACTIVE_TRAINING_STATUSES:
            return _serialize_training_job(current_job), False

        current_job = TrainingJobState(
            id=f"train_{uuid4().hex[:12]}",
            status="queued",
            created_at=now_iso(),
        )
        TRAINING_JOB_STATE = current_job

    thread = Thread(
        target=_run_training_job,
        args=(current_job.id,),
        daemon=True,
        name=f"training-job-{current_job.id}",
    )
    thread.start()
    return _serialize_training_job(current_job), True


def read_image_upload(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(BytesIO(upload.file.read())).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Nao foi possivel abrir a imagem enviada.") from exc


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


def load_image_and_mask(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    bundle = annotation_bundle(sample_id)
    image = np.array(Image.open(bundle["image"]).convert("RGB"))
    mask = np.array(Image.open(bundle["mask"]).convert("L"))
    return image, mask


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
        raise HTTPException(status_code=400, detail="Nao ha pixels anotados suficientes para treino.")
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


def evaluate_classifier(classifier: RandomForestClassifier, records: list[dict]) -> dict:
    if not records:
        return {
            "pixel_accuracy": None,
            "mean_iou": None,
            "per_class_iou": {meta["slug"]: None for meta in class_catalog()},
        }
    all_true = []
    all_pred = []
    for record in records:
        image_rgb, mask = load_image_and_mask(record["id"])
        pred = classifier.predict(build_features(image_rgb)).reshape(mask.shape)
        all_true.append(mask.reshape(-1))
        all_pred.append(pred.reshape(-1))
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def train_latest_model() -> dict:
    with tracked_task(kind="training", label="Treino do modelo", metadata={"model": "RandomForest"}) as task:
        task.update(phase="Lendo anotacoes")
        records = list_annotation_records()
        if len(records) < 2:
            raise HTTPException(
                status_code=400,
                detail="Envie pelo menos 2 itens para a galeria antes de treinar o modelo.",
            )

        task.update(phase="Organizando dataset", metadata={"annotation_count": len(records)})
        split_map = rebuild_dataset_split(records)
        records_by_id = {record["id"]: record for record in records}
        train_records = [records_by_id[sample_id] for sample_id in split_map["train"]]
        val_records = [records_by_id[sample_id] for sample_id in split_map["val"]]
        test_records = [records_by_id[sample_id] for sample_id in split_map["test"]]

        task.update(phase="Extraindo features", metadata={"train_images": len(train_records)})
        feature_batches = []
        label_batches = []
        for record in train_records:
            image_rgb, mask = load_image_and_mask(record["id"])
            features, labels = sample_training_pixels(image_rgb, mask)
            feature_batches.append(features)
            label_batches.append(labels)

        train_x = np.vstack(feature_batches)
        train_y = np.concatenate(label_batches)

        task.update(
            phase="Treinando RandomForest",
            metadata={"train_samples": int(len(train_y))},
        )
        classifier = RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )
        classifier.fit(train_x, train_y)

        ensure_directory(MODELS_DIR)
        trained_at = now_iso()
        joblib.dump({"classifier": classifier, "trained_at": trained_at}, MODEL_PATH)

        task.update(phase="Avaliando e salvando relatorio")
        payload = {
            "trained_at": trained_at,
            "splits": {key: len(value) for key, value in split_map.items()},
            "dataset_ids": split_map,
            "train_samples": int(len(train_y)),
            "train_metrics": evaluate_classifier(classifier, train_records),
            "val_metrics": evaluate_classifier(classifier, val_records),
            "test_metrics": evaluate_classifier(classifier, test_records),
            "classes": class_catalog(),
            "model_path": str(MODEL_PATH),
        }
        ensure_directory(TRAINING_DIR)
        history_path = TRAINING_DIR / f"report_{trained_at.replace(':', '-').replace('.', '-')}.json"
        write_json(history_path, payload)
        write_json(TRAINING_DIR / "latest.json", payload)
        task.update(phase="Concluido")
        return payload


def load_latest_model() -> LoadedModel | None:
    if not MODEL_PATH.exists():
        return None
    payload = joblib.load(MODEL_PATH)
    return LoadedModel(classifier=payload["classifier"], trained_at=payload["trained_at"])


def ensure_model_ready() -> LoadedModel:
    model = load_latest_model()
    if model is not None:
        return model
    raise HTTPException(
        status_code=400,
        detail="Ainda nao existe modelo treinado. Rode o treino antes de analisar novas imagens.",
    )


def calculate_inference_payload(class_mask: np.ndarray) -> dict:
    distribution = compute_pixel_distribution(class_mask)
    counts = distribution["counts"]
    total = distribution["total_pixels"]
    coffee_metrics = compute_coffee_metrics(counts, total)
    return {
        **distribution,
        **coffee_metrics,
    }


def run_inference(image_file: UploadFile) -> dict:
    with tracked_task(
        kind="inference",
        label="Inferencia em imagem",
        metadata={"filename": image_file.filename or "imagem.png"},
    ) as task:
        task.update(phase="Carregando modelo")
        model = ensure_model_ready()
        task.update(phase="Lendo imagem")
        source = read_image_upload(image_file)
        image_rgb = np.array(source)
        features = build_features(image_rgb)

        task.update(phase="Segmentando imagem")
        prediction = model.classifier.predict(features).reshape(image_rgb.shape[:2]).astype(np.uint8)

        color_mask = build_color_mask(prediction)
        overlay = build_overlay(image_rgb, prediction)
        run_id = make_asset_id("infer")
        bundle = inference_bundle(run_id)
        ensure_directory(bundle["base"])

        task.update(phase="Salvando resultados", metadata={"run_id": run_id})
        source.save(bundle["image"])
        Image.fromarray(prediction, mode="L").save(bundle["mask"])
        Image.fromarray(color_mask).save(bundle["color_mask"])
        Image.fromarray(overlay).save(bundle["overlay"])

        metrics = calculate_inference_payload(prediction)
        payload = {
            "id": run_id,
            "storage_key": run_id,
            "original_filename": image_file.filename or "imagem.png",
            "created_at": now_iso(),
            "trained_at": model.trained_at,
            "width": source.width,
            "height": source.height,
            **metrics,
        }
        write_json(bundle["metadata"], payload)
        task.update(phase="Concluido")
        return {
            **payload,
            "image_url": f"/storage/inferences/{run_id}/input.png",
            "image_preview_url": storage_url(ensure_preview_image(bundle["image"], bundle["image_preview"])),
            "mask_url": f"/storage/inferences/{run_id}/mask.png",
            "color_mask_url": f"/storage/inferences/{run_id}/colored_mask.png",
            "overlay_url": f"/storage/inferences/{run_id}/overlay.png",
            "overlay_preview_url": storage_url(ensure_preview_image(bundle["overlay"], bundle["overlay_preview"])),
        }


def delete_inference(run_id: str) -> dict:
    with tracked_task(
        kind="inference_delete",
        label="Excluir inferencia",
        metadata={"run_id": run_id},
    ) as task:
        bundle = inference_bundle(run_id)
        metadata_path = bundle["metadata"]
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Inferencia nao encontrada.")

        payload = read_json(metadata_path)
        task.update(
            phase="Removendo arquivos da inferencia",
            metadata={"trained_at": payload.get("trained_at")},
        )
        shutil.rmtree(bundle["base"], ignore_errors=True)
        task.update(phase="Concluido")
        return payload


def build_model_download_filename(trained_at: str | None = None) -> str:
    safe_suffix = "latest"
    if trained_at:
        safe_suffix = trained_at.replace(":", "-").replace(".", "-")
    return f"facilita-coffee-model_{safe_suffix}.joblib"


def training_status() -> dict:
    report = latest_training_report()
    has_model = MODEL_PATH.exists()
    with TRAINING_JOB_LOCK:
        current_job = TRAINING_JOB_STATE

    return {
        "has_model": has_model,
        "latest_report": report,
        "inference_runs": len(list_inference_records()),
        "download_url": "/api/training/model" if has_model else None,
        "model_filename": build_model_download_filename(report["trained_at"] if report else None) if has_model else None,
        "model_size_bytes": MODEL_PATH.stat().st_size if has_model else None,
        "job": _serialize_training_job(current_job),
    }

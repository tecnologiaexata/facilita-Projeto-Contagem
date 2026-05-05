from io import BytesIO
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP, INFERENCE_PROVIDER, WORKER_DEFAULT_YOLO_MODEL
from app.logging_utils import get_logger
from app.services.annotation import (
    build_class_mask_from_txt,
    build_color_mask,
    build_overlay,
    compute_pixel_distribution,
    decode_mask,
)
from app.services.blob_store import (
    blob_access,
    download_blob_bytes,
    guess_filename_from_reference,
    is_blob_reference,
    upload_blob_bytes,
    upload_json_blob,
)
from app.services.cvat import export_cvat_for_mask
from app.services.gpu_runtime import normalize_requested_device, require_gpu_device, torch_runtime_info
from app.services.modeling import build_features, calculate_inference_payload, compute_metrics
from app.services.yolo_segmentation import (
    build_yolo_annotation_text_from_mask,
    build_training_summary,
    ensure_ultralytics_available,
    evaluate_yolo_model_on_samples,
    export_samples_to_yolo_dataset,
    predict_sample_class_mask,
    resolve_prediction_imgsz,
    resolve_training_params,
    resolve_training_runtime_params,
    resolve_yolo_model_reference,
    train_yolo_segmentation,
)
from app.services.remote_assets import fetch_remote_image, fetch_remote_text
from app.services.roboflow_inference import run_roboflow_inference
from app.services.storage import build_split_map, class_catalog, make_asset_id, now_iso


logger = get_logger("facilita.worker.jobs")


def _payload_value(payload: dict, snake_key: str, camel_key: str):
    if snake_key in payload:
        return payload[snake_key]
    return payload.get(camel_key)


def _context_value(context: dict, snake_key: str, camel_key: str):
    if snake_key in context:
        return context[snake_key]
    return context.get(camel_key)


def _source_reference_label(source) -> str:
    if isinstance(source, dict):
        return str(source.get("download_url") or source.get("url") or source.get("pathname") or "dict-source")
    return str(source or "")


def _asset_reference(source: dict | None) -> str | None:
    if not isinstance(source, dict):
        return None
    return source.get("download_url") or source.get("url") or source.get("pathname")


def _asset_expected_size(source: dict | None) -> int | None:
    if not isinstance(source, dict):
        return None

    value = source.get("size")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _report_progress(report_progress, stage: str, detail: str | None = None, **details) -> None:
    if callable(report_progress):
        report_progress(stage, detail, details=details or None)


def _load_oriented_image(raw: bytes, *, convert_mode: str, reference: str, filename: str) -> Image.Image:
    image = Image.open(BytesIO(raw))
    original_size = image.size
    image = ImageOps.exif_transpose(image)
    if image.size != original_size:
        logger.info(
            "Orientacao EXIF aplicada na imagem: reference=%s filename=%s from=%s to=%s",
            reference,
            filename,
            original_size,
            image.size,
        )
    return image.convert(convert_mode)


def _read_image_source(source, *, fallback_filename: str, convert_mode: str = "RGB") -> tuple[Image.Image, str]:
    if isinstance(source, dict):
        reference = _asset_reference(source)
        if reference:
            logger.info("Baixando imagem do Blob: reference=%s", reference)
            raw = download_blob_bytes(
                reference,
                access=source.get("access") or blob_access(),
                expected_size=_asset_expected_size(source),
            )
            filename = source.get("filename") or source.get("original_filename")
            if not filename:
                filename = guess_filename_from_reference(reference, fallback_filename)
            logger.info("Imagem carregada do Blob: reference=%s filename=%s bytes=%s", reference, filename, len(raw))
            return _load_oriented_image(
                raw,
                convert_mode=convert_mode,
                reference=reference,
                filename=filename,
            ), filename

    reference = str(source or "").strip()
    if not reference:
        raise HTTPException(status_code=400, detail="Referencia de imagem nao informada.")

    if is_blob_reference(reference):
        logger.info("Baixando imagem por referencia Blob: reference=%s", reference)
        raw = download_blob_bytes(reference, access=blob_access())
        logger.info("Imagem carregada por referencia Blob: reference=%s bytes=%s", reference, len(raw))
        return (
            _load_oriented_image(
                raw,
                convert_mode=convert_mode,
                reference=reference,
                filename=guess_filename_from_reference(reference, fallback_filename),
            ),
            guess_filename_from_reference(reference, fallback_filename),
        )

    logger.info("Baixando imagem remota externa: reference=%s", reference)
    image, filename = fetch_remote_image(reference, fallback_filename=fallback_filename)
    logger.info("Imagem remota carregada: reference=%s filename=%s size=%sx%s", reference, filename, image.width, image.height)
    return image.convert(convert_mode), filename


def _read_text_source(source, *, fallback_filename: str) -> tuple[str, str]:
    if isinstance(source, dict):
        reference = _asset_reference(source)
        if reference:
            logger.info("Baixando anotacao TXT do Blob: reference=%s", reference)
            raw = download_blob_bytes(
                reference,
                access=source.get("access") or blob_access(),
                expected_size=_asset_expected_size(source),
            )
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    content = raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
                if content.strip():
                    filename = source.get("filename") or guess_filename_from_reference(reference, fallback_filename)
                    logger.info(
                        "TXT de anotacao carregado do Blob: reference=%s filename=%s encoding=%s bytes=%s",
                        reference,
                        filename,
                        encoding,
                        len(raw),
                    )
                    return content, filename
            raise HTTPException(status_code=400, detail="Nao foi possivel interpretar o TXT de anotacao.")

    reference = str(source or "").strip()
    if not reference:
        raise HTTPException(status_code=400, detail="Referencia de texto nao informada.")

    if is_blob_reference(reference):
        logger.info("Baixando anotacao TXT por referencia Blob: reference=%s", reference)
        raw = download_blob_bytes(reference, access=blob_access())
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            if content.strip():
                logger.info(
                    "TXT de anotacao carregado por referencia Blob: reference=%s encoding=%s bytes=%s",
                    reference,
                    encoding,
                    len(raw),
                )
                return content, guess_filename_from_reference(reference, fallback_filename)
        raise HTTPException(status_code=400, detail="Nao foi possivel interpretar o TXT de anotacao.")

    logger.info("Baixando anotacao TXT remota externa: reference=%s", reference)
    return fetch_remote_text(reference, fallback_filename=fallback_filename)


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _safe_dataset_stem(filename: str | None, fallback: str) -> str:
    value = os.path.splitext(str(filename or "").strip())[0]
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or fallback


def _dataset_image_filename(original_filename: str | None, fallback_stem: str) -> str:
    suffix = os.path.splitext(str(original_filename or "").strip())[1].lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"
    return f"{_safe_dataset_stem(original_filename, fallback_stem)}{suffix}"


def _encode_image_for_filename(image: Image.Image, filename: str) -> bytes:
    suffix = os.path.splitext(str(filename or "").strip())[1].lower()
    buffer = BytesIO()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
    elif suffix == ".webp":
        image.save(buffer, format="WEBP", quality=95)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _annotated_class_slugs_from_mask(class_mask: np.ndarray) -> list[str]:
    return [
        CLASS_MAP[class_id]["slug"]
        for class_id in ANNOTATED_CLASS_IDS
        if bool(np.any(class_mask == class_id))
    ]


def _build_annotation_result(
    *,
    source_image: Image.Image,
    original_filename: str,
    class_mask: np.ndarray,
    output_prefix: str,
    sample_id: str,
    request_id: str | None = None,
    annotation_text: str | None = None,
    annotation_meta: dict | None = None,
    source_type: str,
    annotation_source: str,
    source_reference: dict | None = None,
) -> dict:
    logger.info(
        "Gerando artefatos da amostra anotada: sample_id=%s filename=%s output_prefix=%s source_type=%s annotation_source=%s",
        sample_id,
        original_filename,
        output_prefix,
        source_type,
        annotation_source,
    )
    image_bytes = _encode_png(source_image)
    mask_bytes = _encode_png(Image.fromarray(class_mask, mode="L"))
    color_mask = build_color_mask(class_mask)
    color_mask_bytes = _encode_png(Image.fromarray(color_mask))
    overlay = build_overlay(np.array(source_image), class_mask)
    overlay_bytes = _encode_png(Image.fromarray(overlay))
    cvat_bytes = export_cvat_for_mask(
        sample_id=sample_id,
        original_filename=original_filename,
        class_mask=class_mask,
        width=source_image.width,
        height=source_image.height,
    )

    assets = {
        "image": upload_blob_bytes(f"{output_prefix}/original.png", image_bytes, content_type="image/png"),
        "mask": upload_blob_bytes(f"{output_prefix}/mask.png", mask_bytes, content_type="image/png"),
        "color_mask": upload_blob_bytes(
            f"{output_prefix}/color-mask.png",
            color_mask_bytes,
            content_type="image/png",
        ),
        "overlay": upload_blob_bytes(f"{output_prefix}/overlay.png", overlay_bytes, content_type="image/png"),
        "cvat_xml": upload_blob_bytes(
            f"{output_prefix}/annotation.xml",
            cvat_bytes,
            content_type="application/xml; charset=utf-8",
        ),
    }
    if annotation_text is not None:
        assets["annotation_txt"] = upload_blob_bytes(
            f"{output_prefix}/annotation.txt",
            annotation_text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )

    item = {
        "id": sample_id,
        "request_id": request_id,
        "original_filename": original_filename,
        "source_type": source_type,
        "annotation_source": annotation_source,
        "annotation_format": (annotation_meta or {}).get("annotation_format"),
        "annotation_shape_count": (annotation_meta or {}).get("annotation_shape_count"),
        "annotation_classes": (annotation_meta or {}).get("annotation_classes") or [],
        "annotation_numeric_map": (annotation_meta or {}).get("annotation_numeric_map") or {},
        "width": source_image.width,
        "height": source_image.height,
        "pixel_stats": compute_pixel_distribution(class_mask),
        "assets": assets,
        "metadata": {
            "request_id": request_id,
            "source_reference": source_reference or {},
            "blob_access": blob_access(),
        },
    }
    item["assets"]["metadata_json"] = upload_json_blob(f"{output_prefix}/metadata.json", item)
    logger.info(
        "Amostra anotada pronta: sample_id=%s width=%s height=%s assets=%s",
        sample_id,
        source_image.width,
        source_image.height,
        sorted(item["assets"].keys()),
    )
    return item


def _process_gallery_import(payload: dict, context: dict, report_progress=None) -> dict:
    image_source = _payload_value(payload, "image_url", "imageUrl")
    annotation_txt_source = _payload_value(payload, "annotation_txt_url", "annotationTxtUrl")
    mask_image_source = _payload_value(payload, "mask_image_url", "maskImageUrl")
    if not image_source or (not annotation_txt_source and not mask_image_source):
        raise HTTPException(
            status_code=400,
            detail="Job de galeria precisa de image_url e annotation_txt_url ou mask_image_url.",
        )

    output = context.get("output") or {}
    sample_id = _context_value(output, "sample_id", "sampleId") or _payload_value(payload, "sample_id", "sampleId")
    sample_id = sample_id or make_asset_id("annot")
    output_prefix = _context_value(output, "prefix", "prefix") or f"annotation-samples/{sample_id}"
    logger.info(
        "Processando job de galeria: sample_id=%s image_source=%s annotation_txt_source=%s mask_source=%s",
        sample_id,
        _source_reference_label(image_source),
        _source_reference_label(annotation_txt_source),
        _source_reference_label(mask_image_source),
    )

    _report_progress(
        report_progress,
        "loading_gallery_image",
        "Carregando imagem principal da amostra.",
        sample_id=sample_id,
        output_prefix=output_prefix,
    )
    source_image, original_filename = _read_image_source(
        image_source,
        fallback_filename="imagem-remota.png",
        convert_mode="RGB",
    )

    if annotation_txt_source:
        _report_progress(
            report_progress,
            "loading_annotation",
            "Baixando arquivo TXT da anotacao.",
            sample_id=sample_id,
        )
        annotation_text, _ = _read_text_source(annotation_txt_source, fallback_filename="annotation.txt")
        _report_progress(
            report_progress,
            "building_mask",
            "Convertendo TXT para mascara da amostra.",
            sample_id=sample_id,
        )
        class_mask, annotation_meta = build_class_mask_from_txt(
            annotation_text,
            source_image.width,
            source_image.height,
        )
        logger.info(
            "Anotacao TXT convertida para mascara: sample_id=%s classes=%s shapes=%s",
            sample_id,
            (annotation_meta or {}).get("annotation_classes"),
            (annotation_meta or {}).get("annotation_shape_count"),
        )
        _report_progress(
            report_progress,
            "uploading_annotation_artifacts",
            "Enviando artefatos normalizados da amostra para o Blob.",
            sample_id=sample_id,
        )
        return {
            "item": _build_annotation_result(
                source_image=source_image,
                original_filename=original_filename,
                class_mask=class_mask,
                output_prefix=output_prefix,
                sample_id=sample_id,
                request_id=_payload_value(payload, "request_id", "requestId"),
                annotation_text=annotation_text,
                annotation_meta=annotation_meta,
                source_type="external_txt",
                annotation_source="third_party_txt",
                source_reference={
                    "image_url": image_source,
                    "annotation_txt_url": annotation_txt_source,
                },
            )
        }

    _report_progress(
        report_progress,
        "loading_annotation",
        "Baixando mascara da amostra.",
        sample_id=sample_id,
    )
    mask_image, _ = _read_image_source(mask_image_source, fallback_filename="mask.png", convert_mode="RGB")
    if mask_image.size != source_image.size:
        logger.info(
            "Mascara com tamanho diferente; redimensionando: sample_id=%s from=%s to=%s",
            sample_id,
            mask_image.size,
            source_image.size,
        )
        mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)
    _report_progress(
        report_progress,
        "building_mask",
        "Decodificando mascara enviada.",
        sample_id=sample_id,
    )
    class_mask = decode_mask(mask_image)
    logger.info("Mascara recebida e decodificada: sample_id=%s", sample_id)
    _report_progress(
        report_progress,
        "uploading_annotation_artifacts",
        "Enviando artefatos normalizados da amostra para o Blob.",
        sample_id=sample_id,
    )
    return {
        "item": _build_annotation_result(
            source_image=source_image,
            original_filename=original_filename,
            class_mask=class_mask,
            output_prefix=output_prefix,
            sample_id=sample_id,
            request_id=_payload_value(payload, "request_id", "requestId"),
            annotation_text=None,
            annotation_meta={
                "annotation_format": "mask_image",
                "annotation_shape_count": 0,
                "annotation_classes": _annotated_class_slugs_from_mask(class_mask),
                "annotation_numeric_map": {},
            },
            source_type="external_mask_image",
            annotation_source="mask_image",
            source_reference={
                "image_url": image_source,
                "mask_image_url": mask_image_source,
            },
        )
    }


def _load_training_sample_arrays(sample: dict, *, report_progress=None, sample_index: int | None = None, sample_count: int | None = None) -> dict:
    assets = sample.get("assets") or {}
    image_asset = assets.get("image")
    mask_asset = assets.get("mask")
    if not image_asset or not mask_asset:
        raise HTTPException(status_code=400, detail=f"A amostra {sample.get('id')} nao possui assets completos.")

    progress_prefix = ""
    if sample_index is not None and sample_count is not None:
        progress_prefix = f"Amostra {sample_index}/{sample_count}: "

    _report_progress(
        report_progress,
        "loading_samples",
        f"{progress_prefix}baixando assets da amostra {sample.get('id')}.",
        sample_id=sample.get("id"),
        sample_index=sample_index,
        sample_count=sample_count,
    )
    logger.info("Carregando amostra para treino: sample_id=%s", sample.get("id"))
    image_bytes = download_blob_bytes(
        _asset_reference(image_asset),
        access=image_asset.get("access") or blob_access(),
        expected_size=_asset_expected_size(image_asset),
    )
    mask_bytes = download_blob_bytes(
        _asset_reference(mask_asset),
        access=mask_asset.get("access") or blob_access(),
        expected_size=_asset_expected_size(mask_asset),
    )
    image_rgb = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
    logger.info(
        "Amostra carregada para treino: sample_id=%s image_shape=%s mask_shape=%s",
        sample.get("id"),
        image_rgb.shape,
        mask.shape,
    )
    _report_progress(
        report_progress,
        "loading_samples",
        f"{progress_prefix}amostra {sample.get('id')} carregada para treino.",
        sample_id=sample.get("id"),
        sample_index=sample_index,
        sample_count=sample_count,
        image_shape=list(image_rgb.shape),
        mask_shape=list(mask.shape),
    )
    return {
        "id": sample.get("id"),
        "created_at": sample.get("createdAt") or sample.get("created_at") or now_iso(),
        "image_rgb": image_rgb,
        "mask": mask,
    }


def _empty_metric_payload() -> dict:
    return {
        "pixel_accuracy": None,
        "mean_iou": None,
        "per_class_iou": {meta["slug"]: None for meta in class_catalog()},
    }


def _evaluate_classifier_on_loaded_samples(
    classifier,
    samples: list[dict],
) -> dict:
    if not samples:
        return _empty_metric_payload()

    logger.info("Avaliando classificador em %s amostras.", len(samples))
    all_true = []
    all_pred = []
    for sample in samples:
        pred = classifier.predict(build_features(sample["image_rgb"])).reshape(sample["mask"].shape)
        all_true.append(sample["mask"].reshape(-1))
        all_pred.append(pred.reshape(-1))
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def _process_training(context: dict, report_progress=None) -> dict:
    ensure_ultralytics_available()
    dataset = context.get("dataset") or {}
    raw_samples = dataset.get("samples") or []
    if len(raw_samples) < 2:
        raise HTTPException(
            status_code=400,
            detail="Sao necessarias pelo menos 2 amostras anotadas para treinar o modelo.",
        )

    output = context.get("output") or {}
    params = resolve_training_params(context)
    params["device"] = require_gpu_device(params.get("device"), operation="Treino YOLO Segmentation")
    training_run_id = _context_value(output, "training_run_id", "trainingRunId") or make_asset_id("train")
    output_prefix = _context_value(output, "prefix", "prefix") or f"training-runs/{training_run_id}"
    logger.info(
        "Processando job de treino YOLO: training_run_id=%s sample_count=%s output_prefix=%s",
        training_run_id,
        len(raw_samples),
        output_prefix,
    )

    _report_progress(
        report_progress,
        "loading_samples",
        f"Carregando {len(raw_samples)} amostras para o treino YOLO.",
        training_run_id=training_run_id,
        sample_count=len(raw_samples),
    )
    loaded_samples = [
        _load_training_sample_arrays(
            sample,
            report_progress=report_progress,
            sample_index=index,
            sample_count=len(raw_samples),
        )
        for index, sample in enumerate(raw_samples, start=1)
    ]
    split_map = build_split_map(
        [{"id": sample["id"], "created_at": sample["created_at"]} for sample in loaded_samples]
    )
    loaded_by_id = {sample["id"]: sample for sample in loaded_samples}
    train_records = [loaded_by_id[sample_id] for sample_id in split_map["train"]]
    val_records = [loaded_by_id[sample_id] for sample_id in split_map["val"]]
    test_records = [loaded_by_id[sample_id] for sample_id in split_map["test"]]
    params = resolve_training_runtime_params(params, loaded_samples)
    _report_progress(
        report_progress,
        "preparing_yolo_dataset",
        "Convertendo amostras normalizadas para dataset YOLO Segmentation.",
        training_run_id=training_run_id,
        params=params,
    )

    with tempfile.TemporaryDirectory(prefix=f"facilita-yolo-{training_run_id}-") as workdir:
        dataset_paths = export_samples_to_yolo_dataset(
            loaded_samples=loaded_samples,
            split_map=split_map,
            output_dir=workdir,
            params=params,
        )
        _report_progress(
            report_progress,
            "training_model",
            "Treinando YOLO Segmentation.",
            training_run_id=training_run_id,
            params=params,
            data_yaml=dataset_paths["data_yaml"],
        )
        train_artifacts = train_yolo_segmentation(
            data_yaml=dataset_paths["data_yaml"],
            output_dir=workdir,
            run_name=training_run_id,
            params=params,
            progress_callback=report_progress,
            training_run_id=training_run_id,
        )

        _report_progress(
            report_progress,
            "evaluating_model",
            "Avaliando o melhor checkpoint do YOLO nas particoes train/val/test.",
            training_run_id=training_run_id,
        )
        train_metrics = evaluate_yolo_model_on_samples(train_artifacts["best_model_path"], train_records, params=params)
        val_metrics = evaluate_yolo_model_on_samples(train_artifacts["best_model_path"], val_records, params=params)
        test_metrics = evaluate_yolo_model_on_samples(train_artifacts["best_model_path"], test_records, params=params)

        summary = build_training_summary(
            training_run_id=training_run_id,
            train_artifacts=train_artifacts,
            params=params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            split_map=split_map,
        )

        trained_at = now_iso()
        best_model_bytes = Path(train_artifacts["best_model_path"]).read_bytes()
        last_model_bytes = Path(train_artifacts["last_model_path"]).read_bytes() if train_artifacts.get("last_model_path") and Path(train_artifacts["last_model_path"]).exists() else None

        item = {
            "id": training_run_id,
            "trained_at": trained_at,
            "train_samples": int(len(train_records)),
            "splits": {key: len(value) for key, value in split_map.items()},
            "dataset_ids": split_map,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "classes": class_catalog(),
            "assets": {
                "model": upload_blob_bytes(
                    f"{output_prefix}/best.pt",
                    best_model_bytes,
                    content_type="application/octet-stream",
                ),
                "report_json": upload_json_blob(f"{output_prefix}/report.json", summary),
                "report_md": upload_blob_bytes(
                    f"{output_prefix}/report.md",
                    summary["markdown"].encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                ),
            },
            "metadata": {
                "model": "YOLO Segmentation",
                "task": "segment",
                "base_model": params["model"],
                "imgsz": params["imgsz"],
                "native_resolution": params.get("native_resolution"),
                "resolution_mode": params.get("resolution_mode"),
                "tile_enabled": params.get("tile_enabled"),
                "tile_size": params.get("tile_size"),
                "tile_overlap": params.get("tile_overlap"),
                "tile_count_estimate": params.get("tile_count_estimate"),
                "dataset_tiles": dataset_paths.get("exported_counts"),
                "epochs": params["epochs"],
                "batch": params["batch"],
                "batch_mode": params.get("batch_mode"),
                "patience": params["patience"],
                "optimizer": params["optimizer"],
                "conf": params["conf"],
                "iou": params["iou"],
                "mask_threshold": params["mask_threshold"],
                "plant_inference_mode": "exclusion",
                "summary": summary.get("executive_summary"),
            },
        }
        if last_model_bytes is not None:
            item["assets"]["last_model"] = upload_blob_bytes(
                f"{output_prefix}/last.pt",
                last_model_bytes,
                content_type="application/octet-stream",
            )
        if train_artifacts.get("results_csv") and Path(train_artifacts["results_csv"]).exists():
            item["assets"]["history_csv"] = upload_blob_bytes(
                f"{output_prefix}/history.csv",
                Path(train_artifacts["results_csv"]).read_bytes(),
                content_type="text/csv; charset=utf-8",
            )
        if train_artifacts.get("results_png") and Path(train_artifacts["results_png"]).exists():
            item["assets"]["results_plot"] = upload_blob_bytes(
                f"{output_prefix}/results.png",
                Path(train_artifacts["results_png"]).read_bytes(),
                content_type="image/png",
            )

    logger.info(
        "Treino YOLO concluido: training_run_id=%s best_model=%s",
        training_run_id,
        item["assets"]["model"],
    )
    return {"item": item}

def _load_model_from_context(context: dict):
    ensure_ultralytics_available()
    from ultralytics import YOLO

    model = context.get("model") or {}
    model_asset = model.get("asset")
    local_model_reference = (
        model.get("local_path")
        or model.get("localPath")
        or model.get("path")
        or model.get("base_model")
        or model.get("baseModel")
        or WORKER_DEFAULT_YOLO_MODEL
    )
    if not model_asset:
        resolved_local_model = resolve_yolo_model_reference(local_model_reference)
        if Path(resolved_local_model).exists():
            logger.info("Carregando modelo YOLO local para inferencia: path=%s", resolved_local_model)
            yolo = YOLO(resolved_local_model)
            return yolo, None, "local-default", None
        raise HTTPException(
            status_code=400,
            detail=(
                "Nao existe modelo ativo no control plane para executar a inferencia, "
                "e nenhum WORKER_DEFAULT_YOLO_MODEL local foi encontrado."
            ),
        )
    logger.info("Baixando modelo ativo YOLO para inferencia: model_id=%s", model.get("id"))
    model_reference = _asset_reference(model_asset)
    model_bytes = download_blob_bytes(
        model_reference,
        access=model_asset.get("access") or blob_access(),
        expected_size=_asset_expected_size(model_asset),
    )
    model_filename = model_asset.get("filename") or guess_filename_from_reference(model_reference, "model.pt")
    suffix = os.path.splitext(model_filename)[1] or ".pt"
    temp_file = tempfile.NamedTemporaryFile(prefix="facilita-model-", suffix=suffix, delete=False)
    temp_file.write(model_bytes)
    temp_file.flush()
    temp_file.close()
    yolo = YOLO(temp_file.name)
    logger.info("Modelo YOLO carregado para inferencia: model_id=%s trained_at=%s", model.get("id"), model.get("trained_at"))
    return yolo, model.get("trained_at"), model.get("id"), temp_file.name


def _inference_provider_from_payload(payload: dict) -> str:
    raw_provider = (
        _payload_value(payload, "inference_provider", "inferenceProvider")
        or _payload_value(payload, "provider", "provider")
        or INFERENCE_PROVIDER
    )
    provider = str(raw_provider or "local_yolo").strip().lower().replace("-", "_")
    if provider in {"yolo", "local", "local_yolo"}:
        return "local_yolo"
    if provider == "roboflow":
        return "roboflow"
    raise HTTPException(status_code=400, detail=f"Provider de inferencia nao suportado: {provider}.")


def _process_roboflow_inference(payload: dict, context: dict, report_progress=None) -> dict:
    image_source = _payload_value(payload, "image_url", "imageUrl")
    if not image_source:
        raise HTTPException(status_code=400, detail="Job de inferencia precisa de image_url.")

    output = context.get("output") or {}
    inference_run_id = _context_value(output, "inference_run_id", "inferenceRunId") or make_asset_id("infer")
    output_prefix = _context_value(output, "prefix", "prefix") or f"inference-runs/{inference_run_id}"
    confidence = _payload_value(payload, "confidence", "confidence")
    logger.info(
        "Processando job de inferencia Roboflow: inference_run_id=%s image_source=%s output_prefix=%s confidence=%s",
        inference_run_id,
        _source_reference_label(image_source),
        output_prefix,
        confidence,
    )

    _report_progress(
        report_progress,
        "loading_input",
        "Carregando imagem de entrada para inferencia Roboflow.",
        inference_run_id=inference_run_id,
    )
    source_image, original_filename = _read_image_source(
        image_source,
        fallback_filename="imagem-remota.png",
        convert_mode="RGB",
    )
    _report_progress(
        report_progress,
        "predicting_roboflow",
        "Executando Workflow hospedado no Roboflow.",
        inference_run_id=inference_run_id,
        image_shape=[source_image.height, source_image.width, 3],
    )
    roboflow_result = run_roboflow_inference(source_image, confidence=confidence)
    prediction = roboflow_result["class_mask"]
    inference_image = roboflow_result["image"]
    image_rgb = roboflow_result["image_rgb"]
    roboflow_metadata = roboflow_result["metadata"]
    logger.info(
        "Inferencia Roboflow calculada: inference_run_id=%s image_shape=%s output_mode=%s",
        inference_run_id,
        image_rgb.shape,
        roboflow_metadata.get("output_mode"),
    )

    _report_progress(
        report_progress,
        "uploading_inference_artifacts",
        "Enviando artefatos da inferencia Roboflow para o Blob.",
        inference_run_id=inference_run_id,
        output_mode=roboflow_metadata.get("output_mode"),
    )
    color_mask = build_color_mask(prediction)
    overlay = build_overlay(image_rgb, prediction)
    annotation_text = build_yolo_annotation_text_from_mask(prediction)
    dataset_image_filename = _dataset_image_filename(original_filename, inference_run_id)
    dataset_label_filename = f"{os.path.splitext(dataset_image_filename)[0]}.txt"

    image_bytes = _encode_image_for_filename(inference_image, dataset_image_filename)
    mask_bytes = _encode_png(Image.fromarray(prediction, mode="L"))
    color_mask_bytes = _encode_png(Image.fromarray(color_mask))
    overlay_bytes = _encode_png(Image.fromarray(overlay))
    metrics = calculate_inference_payload(prediction)

    item = {
        "id": inference_run_id,
        "training_run_id": None,
        "trained_at": None,
        "original_filename": original_filename,
        "width": inference_image.width,
        "height": inference_image.height,
        "metrics": metrics,
        "assets": {
            "image": upload_blob_bytes(
                f"{output_prefix}/dataset/images/{dataset_image_filename}",
                image_bytes,
            ),
            "mask": upload_blob_bytes(f"{output_prefix}/mask.png", mask_bytes, content_type="image/png"),
            "color_mask": upload_blob_bytes(
                f"{output_prefix}/color-mask.png",
                color_mask_bytes,
                content_type="image/png",
            ),
            "overlay": upload_blob_bytes(
                f"{output_prefix}/overlay.png",
                overlay_bytes,
                content_type="image/png",
            ),
            "annotation_txt": upload_blob_bytes(
                f"{output_prefix}/dataset/labels/{dataset_label_filename}",
                annotation_text.encode("utf-8"),
                content_type="text/plain; charset=utf-8",
            ),
            "roboflow_result_json": upload_json_blob(
                f"{output_prefix}/roboflow-result.json",
                roboflow_result["raw_result"],
            ),
        },
        "metadata": {
            "source_reference": {"image_url": image_source},
            "blob_access": blob_access(),
            "task": "segment",
            "provider": "roboflow",
            "roboflow": roboflow_metadata,
            "plant_inference_mode": "exclusion",
            "dataset_export": {
                "image_path": f"dataset/images/{dataset_image_filename}",
                "label_path": f"dataset/labels/{dataset_label_filename}",
            },
        },
    }
    item["assets"]["result_json"] = upload_json_blob(f"{output_prefix}/result.json", item)
    return {"item": item}


def _process_inference(payload: dict, context: dict, report_progress=None) -> dict:
    image_source = _payload_value(payload, "image_url", "imageUrl")
    if not image_source:
        raise HTTPException(status_code=400, detail="Job de inferencia precisa de image_url.")

    provider = _inference_provider_from_payload(payload)
    if provider == "roboflow":
        return _process_roboflow_inference(payload, context, report_progress=report_progress)

    params = resolve_training_params(context)
    requested_device = normalize_requested_device(params.get("device"))
    runtime = torch_runtime_info()
    if runtime.get("cuda_available"):
        device = requested_device
    else:
        device = "cpu"
        logger.warning(
            "CUDA indisponivel neste worker para inferencia; executando fallback em CPU. requested_device=%s runtime=%s",
            requested_device,
            runtime,
        )
    output = context.get("output") or {}
    inference_run_id = _context_value(output, "inference_run_id", "inferenceRunId") or make_asset_id("infer")
    output_prefix = _context_value(output, "prefix", "prefix") or f"inference-runs/{inference_run_id}"
    logger.info(
        "Processando job de inferencia YOLO: inference_run_id=%s image_source=%s output_prefix=%s",
        inference_run_id,
        _source_reference_label(image_source),
        output_prefix,
    )

    _report_progress(
        report_progress,
        "loading_model",
        "Baixando modelo ativo YOLO para executar a inferencia.",
        inference_run_id=inference_run_id,
    )
    yolo_model, trained_at, training_run_id, temp_model_path = _load_model_from_context(context)
    try:
        _report_progress(
            report_progress,
            "loading_input",
            "Carregando imagem de entrada para inferencia.",
            inference_run_id=inference_run_id,
            training_run_id=training_run_id,
        )
        source_image, original_filename = _read_image_source(
            image_source,
            fallback_filename="imagem-remota.png",
            convert_mode="RGB",
        )
        image_rgb = np.array(source_image)
        _report_progress(
            report_progress,
            "predicting",
            "Executando predicao YOLO segmentation.",
            inference_run_id=inference_run_id,
            training_run_id=training_run_id,
            image_shape=list(image_rgb.shape),
        )
        predict_imgsz = resolve_prediction_imgsz(params, image_rgb.shape[:2])
        prediction = predict_sample_class_mask(yolo_model, image_rgb, params=params, device=device)
        logger.info(
            "Inferencia YOLO calculada: inference_run_id=%s image_shape=%s model_id=%s",
            inference_run_id,
            image_rgb.shape,
            training_run_id,
        )
        color_mask = build_color_mask(prediction)
        overlay = build_overlay(image_rgb, prediction)
        annotation_text = build_yolo_annotation_text_from_mask(prediction)
        dataset_image_filename = _dataset_image_filename(original_filename, inference_run_id)
        dataset_label_filename = f"{os.path.splitext(dataset_image_filename)[0]}.txt"

        image_bytes = _encode_image_for_filename(source_image, dataset_image_filename)
        mask_bytes = _encode_png(Image.fromarray(prediction, mode="L"))
        color_mask_bytes = _encode_png(Image.fromarray(color_mask))
        overlay_bytes = _encode_png(Image.fromarray(overlay))
        metrics = calculate_inference_payload(prediction)

        item = {
            "id": inference_run_id,
            "training_run_id": training_run_id,
            "trained_at": trained_at,
            "original_filename": original_filename,
            "width": source_image.width,
            "height": source_image.height,
            "metrics": metrics,
            "assets": {
                "image": upload_blob_bytes(
                    f"{output_prefix}/dataset/images/{dataset_image_filename}",
                    image_bytes,
                ),
                "mask": upload_blob_bytes(f"{output_prefix}/mask.png", mask_bytes, content_type="image/png"),
                "color_mask": upload_blob_bytes(
                    f"{output_prefix}/color-mask.png",
                    color_mask_bytes,
                    content_type="image/png",
                ),
                "overlay": upload_blob_bytes(
                    f"{output_prefix}/overlay.png",
                    overlay_bytes,
                    content_type="image/png",
                ),
                "annotation_txt": upload_blob_bytes(
                    f"{output_prefix}/dataset/labels/{dataset_label_filename}",
                    annotation_text.encode("utf-8"),
                    content_type="text/plain; charset=utf-8",
                ),
            },
            "metadata": {
                "source_reference": {"image_url": image_source},
                "blob_access": blob_access(),
                "task": "segment",
                "provider": "local_yolo",
                "plant_inference_mode": "exclusion",
                "imgsz": predict_imgsz,
                "native_resolution": params.get("native_resolution"),
                "tile_enabled": params.get("tile_enabled"),
                "tile_size": params.get("tile_size"),
                "tile_overlap": params.get("tile_overlap"),
                "dataset_export": {
                    "image_path": f"dataset/images/{dataset_image_filename}",
                    "label_path": f"dataset/labels/{dataset_label_filename}",
                },
            },
        }
        item["assets"]["result_json"] = upload_json_blob(f"{output_prefix}/result.json", item)
        return {"item": item}
    finally:
        if temp_model_path:
            try:
                os.unlink(temp_model_path)
            except OSError:
                pass

def process_control_plane_job(job: dict, context: dict | None = None, report_progress=None) -> dict:
    kind = str(job.get("kind") or "").strip()
    payload = job.get("requestPayload") or job.get("request_payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    context = context or {}
    logger.info(
        "Despachando job para processador local: job_id=%s kind=%s payload_keys=%s context_keys=%s",
        job.get("id"),
        kind,
        sorted(payload.keys()),
        sorted(context.keys()),
    )

    if kind == "gallery_import":
        return _process_gallery_import(payload, context, report_progress=report_progress)

    if kind == "inference":
        return _process_inference(payload, context, report_progress=report_progress)

    if kind == "training":
        return _process_training(context, report_progress=report_progress)

    raise HTTPException(status_code=400, detail=f"Tipo de job nao suportado: {kind or 'desconhecido'}.")

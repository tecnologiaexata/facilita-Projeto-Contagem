from io import BytesIO

import joblib
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP
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
from app.services.modeling import (
    MAX_PIXELS_PER_CLASS,
    build_features,
    calculate_inference_payload,
    compute_metrics,
    sample_training_pixels,
)
from app.services.remote_assets import fetch_remote_image, fetch_remote_text
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
            raw = download_blob_bytes(reference, access=source.get("access") or blob_access())
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
            raw = download_blob_bytes(reference, access=source.get("access") or blob_access())
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


def _process_gallery_import(payload: dict, context: dict) -> dict:
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

    source_image, original_filename = _read_image_source(
        image_source,
        fallback_filename="imagem-remota.png",
        convert_mode="RGB",
    )

    if annotation_txt_source:
        annotation_text, _ = _read_text_source(annotation_txt_source, fallback_filename="annotation.txt")
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

    mask_image, _ = _read_image_source(mask_image_source, fallback_filename="mask.png", convert_mode="RGB")
    if mask_image.size != source_image.size:
        logger.info(
            "Mascara com tamanho diferente; redimensionando: sample_id=%s from=%s to=%s",
            sample_id,
            mask_image.size,
            source_image.size,
        )
        mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)
    class_mask = decode_mask(mask_image)
    logger.info("Mascara recebida e decodificada: sample_id=%s", sample_id)
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


def _load_training_sample_arrays(sample: dict) -> dict:
    assets = sample.get("assets") or {}
    image_asset = assets.get("image")
    mask_asset = assets.get("mask")
    if not image_asset or not mask_asset:
        raise HTTPException(status_code=400, detail=f"A amostra {sample.get('id')} nao possui assets completos.")

    logger.info("Carregando amostra para treino: sample_id=%s", sample.get("id"))
    image_bytes = download_blob_bytes(
        _asset_reference(image_asset),
        access=image_asset.get("access") or blob_access(),
    )
    mask_bytes = download_blob_bytes(
        _asset_reference(mask_asset),
        access=mask_asset.get("access") or blob_access(),
    )
    image_rgb = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
    logger.info(
        "Amostra carregada para treino: sample_id=%s image_shape=%s mask_shape=%s",
        sample.get("id"),
        image_rgb.shape,
        mask.shape,
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
    classifier: RandomForestClassifier,
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


def _process_training(context: dict) -> dict:
    dataset = context.get("dataset") or {}
    raw_samples = dataset.get("samples") or []
    if len(raw_samples) < 2:
        raise HTTPException(
            status_code=400,
            detail="Sao necessarias pelo menos 2 amostras anotadas para treinar o modelo.",
        )

    output = context.get("output") or {}
    training_run_id = _context_value(output, "training_run_id", "trainingRunId") or make_asset_id("train")
    output_prefix = _context_value(output, "prefix", "prefix") or f"training-runs/{training_run_id}"
    logger.info(
        "Processando job de treino: training_run_id=%s sample_count=%s output_prefix=%s",
        training_run_id,
        len(raw_samples),
        output_prefix,
    )

    loaded_samples = [_load_training_sample_arrays(sample) for sample in raw_samples]
    split_map = build_split_map(
        [{"id": sample["id"], "created_at": sample["created_at"]} for sample in loaded_samples]
    )
    loaded_by_id = {sample["id"]: sample for sample in loaded_samples}
    train_records = [loaded_by_id[sample_id] for sample_id in split_map["train"]]
    val_records = [loaded_by_id[sample_id] for sample_id in split_map["val"]]
    test_records = [loaded_by_id[sample_id] for sample_id in split_map["test"]]
    logger.info(
        "Split de treino definido: training_run_id=%s train=%s val=%s test=%s",
        training_run_id,
        len(train_records),
        len(val_records),
        len(test_records),
    )

    feature_batches = []
    label_batches = []
    for sample in train_records:
        features, labels = sample_training_pixels(sample["image_rgb"], sample["mask"])
        feature_batches.append(features)
        label_batches.append(labels)
        logger.info(
            "Pixels amostrados para treino: training_run_id=%s sample_id=%s features=%s labels=%s",
            training_run_id,
            sample["id"],
            features.shape,
            labels.shape,
        )

    train_x = np.vstack(feature_batches)
    train_y = np.concatenate(label_batches)
    logger.info(
        "Dataset de treino consolidado: training_run_id=%s train_x=%s train_y=%s",
        training_run_id,
        train_x.shape,
        train_y.shape,
    )

    classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    logger.info("Treinando RandomForestClassifier: training_run_id=%s", training_run_id)
    classifier.fit(train_x, train_y)
    logger.info("Treinamento concluido: training_run_id=%s", training_run_id)

    trained_at = now_iso()
    model_buffer = BytesIO()
    joblib.dump({"classifier": classifier, "trained_at": trained_at}, model_buffer)

    item = {
        "id": training_run_id,
        "trained_at": trained_at,
        "train_samples": int(len(train_y)),
        "splits": {key: len(value) for key, value in split_map.items()},
        "dataset_ids": split_map,
        "train_metrics": _evaluate_classifier_on_loaded_samples(classifier, train_records),
        "val_metrics": _evaluate_classifier_on_loaded_samples(classifier, val_records),
        "test_metrics": _evaluate_classifier_on_loaded_samples(classifier, test_records),
        "classes": class_catalog(),
        "assets": {
            "model": upload_blob_bytes(
                f"{output_prefix}/model.joblib",
                model_buffer.getvalue(),
                content_type="application/octet-stream",
            ),
        },
        "metadata": {
            "model": "RandomForestClassifier",
            "max_pixels_per_class": MAX_PIXELS_PER_CLASS,
            "rng_seed_hint": 42,
        },
    }
    item["assets"]["report_json"] = upload_json_blob(f"{output_prefix}/report.json", item)
    logger.info(
        "Artefatos de treino enviados ao Blob: training_run_id=%s assets=%s",
        training_run_id,
        sorted(item["assets"].keys()),
    )
    return {"item": item}


def _load_model_from_context(context: dict):
    model = context.get("model") or {}
    model_asset = model.get("asset")
    if not model_asset:
        raise HTTPException(
            status_code=400,
            detail="Nao existe modelo ativo no control plane para executar a inferencia.",
        )
    logger.info("Baixando modelo ativo para inferencia: model_id=%s", model.get("id"))
    model_bytes = download_blob_bytes(
        _asset_reference(model_asset),
        access=model_asset.get("access") or blob_access(),
    )
    payload = joblib.load(BytesIO(model_bytes))
    logger.info("Modelo carregado para inferencia: model_id=%s trained_at=%s", model.get("id"), payload["trained_at"])
    return payload["classifier"], payload["trained_at"], model.get("id")


def _process_inference(payload: dict, context: dict) -> dict:
    image_source = _payload_value(payload, "image_url", "imageUrl")
    if not image_source:
        raise HTTPException(status_code=400, detail="Job de inferencia precisa de image_url.")

    output = context.get("output") or {}
    inference_run_id = _context_value(output, "inference_run_id", "inferenceRunId") or make_asset_id("infer")
    output_prefix = _context_value(output, "prefix", "prefix") or f"inference-runs/{inference_run_id}"
    logger.info(
        "Processando job de inferencia: inference_run_id=%s image_source=%s output_prefix=%s",
        inference_run_id,
        _source_reference_label(image_source),
        output_prefix,
    )

    classifier, trained_at, training_run_id = _load_model_from_context(context)
    source_image, original_filename = _read_image_source(
        image_source,
        fallback_filename="imagem-remota.png",
        convert_mode="RGB",
    )
    image_rgb = np.array(source_image)
    prediction = classifier.predict(build_features(image_rgb)).reshape(image_rgb.shape[:2]).astype(np.uint8)
    logger.info(
        "Inferencia calculada: inference_run_id=%s image_shape=%s model_id=%s",
        inference_run_id,
        image_rgb.shape,
        training_run_id,
    )
    color_mask = build_color_mask(prediction)
    overlay = build_overlay(image_rgb, prediction)

    image_bytes = _encode_png(source_image)
    mask_bytes = _encode_png(Image.fromarray(prediction, mode="L"))
    color_mask_bytes = _encode_png(Image.fromarray(color_mask))
    overlay_bytes = _encode_png(Image.fromarray(overlay))
    metrics = calculate_inference_payload(prediction)
    logger.info(
        "Metricas da inferencia prontas: inference_run_id=%s coffee_percent=%s mapped_percent=%s",
        inference_run_id,
        metrics.get("coffee_percentual_na_imagem"),
        metrics.get("area_mapeada_percentual_na_imagem"),
    )

    item = {
        "id": inference_run_id,
        "training_run_id": training_run_id,
        "trained_at": trained_at,
        "original_filename": original_filename,
        "width": source_image.width,
        "height": source_image.height,
        "metrics": metrics,
        "assets": {
            "image": upload_blob_bytes(f"{output_prefix}/input.png", image_bytes, content_type="image/png"),
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
        },
        "metadata": {
            "source_reference": {"image_url": image_source},
            "blob_access": blob_access(),
        },
    }
    item["assets"]["result_json"] = upload_json_blob(f"{output_prefix}/result.json", item)
    logger.info(
        "Artefatos de inferencia enviados ao Blob: inference_run_id=%s assets=%s",
        inference_run_id,
        sorted(item["assets"].keys()),
    )
    return {"item": item}


def process_control_plane_job(job: dict, context: dict | None = None) -> dict:
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
        return _process_gallery_import(payload, context)

    if kind == "inference":
        return _process_inference(payload, context)

    if kind == "training":
        return _process_training(context)

    raise HTTPException(status_code=400, detail=f"Tipo de job nao suportado: {kind or 'desconhecido'}.")

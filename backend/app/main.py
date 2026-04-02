from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import APP_VERSION, STORAGE_DIR
from app.services.annotation import (
    annotations_summary,
    build_annotation_package,
    delete_annotation,
    save_annotation,
    save_annotation_from_urls,
)
from app.services.control_plane import (
    control_plane_status,
    start_control_plane_heartbeat,
    stop_control_plane_heartbeat,
)
from app.services.modeling import (
    MODEL_PATH,
    build_model_download_filename,
    delete_inference,
    run_inference,
    run_inference_from_url,
    start_training_job,
    training_status,
)
from app.services.storage import (
    class_catalog,
    count_annotation_records,
    ensure_storage,
    load_annotation_record,
    list_annotation_records,
    list_inference_records,
)


ensure_storage()

app = FastAPI(title="Facilita Coffee Counter", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")


class RemoteGalleryPayload(BaseModel):
    image_url: str
    annotation_txt_url: str | None = None
    mask_image_url: str | None = None
    sample_id: str | None = None
    request_id: str | None = None


class RemoteInferencePayload(BaseModel):
    image_url: str


@app.on_event("startup")
def on_startup() -> None:
    start_control_plane_heartbeat()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_control_plane_heartbeat()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "classes": class_catalog(),
        "summary": annotations_summary(),
        "training": training_status(),
        "worker": control_plane_status(),
    }


@app.get("/api/worker")
def worker_meta() -> dict:
    return {"item": control_plane_status()}


def _gallery_payload(offset: int = 0, limit: int | None = None) -> dict:
    total = count_annotation_records()
    records = list_annotation_records(offset=offset, limit=limit)
    return {
        "items": records,
        "total": total,
        "offset": offset,
        "limit": limit if limit is not None else len(records),
    }


@app.get("/api/gallery")
def get_gallery(
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1),
) -> dict:
    return _gallery_payload(offset=offset, limit=limit)


@app.get("/api/annotations")
def get_annotations(
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1),
) -> dict:
    return _gallery_payload(offset=offset, limit=limit)


@app.post("/api/gallery")
def create_gallery_item(
    image: UploadFile = File(...),
    annotation_txt: UploadFile = File(...),
    request_id: str | None = Form(default=None),
) -> dict:
    record = save_annotation(
        image,
        annotation_file=annotation_txt,
        request_id=request_id,
    )
    return {"item": record}


@app.post("/api/annotations")
def create_annotation(
    original_image: UploadFile = File(...),
    mask_image: UploadFile | None = File(default=None),
    annotation_txt: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    request_id: str | None = Form(default=None),
) -> dict:
    record = save_annotation(
        original_image,
        mask_file=mask_image,
        annotation_file=annotation_txt,
        sample_id=sample_id,
        request_id=request_id,
    )
    return {"item": record}


@app.post("/api/gallery/from-url")
def create_gallery_item_from_url(payload: RemoteGalleryPayload) -> dict:
    record = save_annotation_from_urls(
        image_url=payload.image_url,
        annotation_txt_url=payload.annotation_txt_url,
        mask_image_url=payload.mask_image_url,
        sample_id=payload.sample_id,
        request_id=payload.request_id,
    )
    return {"item": record}


@app.post("/api/annotations/from-url")
def create_annotation_from_url(payload: RemoteGalleryPayload) -> dict:
    record = save_annotation_from_urls(
        image_url=payload.image_url,
        annotation_txt_url=payload.annotation_txt_url,
        mask_image_url=payload.mask_image_url,
        sample_id=payload.sample_id,
        request_id=payload.request_id,
    )
    return {"item": record}


@app.get("/api/gallery/{sample_id}")
def get_gallery_item(sample_id: str) -> dict:
    record = load_annotation_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Item da galeria nao encontrado.")
    return {"item": record}


@app.get("/api/annotations/{sample_id}")
def get_annotation(sample_id: str) -> dict:
    record = load_annotation_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Item da galeria nao encontrado.")
    return {"item": record}


@app.delete("/api/gallery/{sample_id}")
def remove_gallery_item(sample_id: str) -> dict:
    return {"item": delete_annotation(sample_id)}


@app.delete("/api/annotations/{sample_id}")
def remove_annotation(sample_id: str) -> dict:
    return {"item": delete_annotation(sample_id)}


@app.get("/api/gallery/{sample_id}/package")
def download_gallery_package(sample_id: str) -> StreamingResponse:
    buffer, filename = build_annotation_package(sample_id)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.get("/api/annotations/{sample_id}/package")
def download_annotation_package(sample_id: str) -> StreamingResponse:
    buffer, filename = build_annotation_package(sample_id)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.get("/api/inferences")
def get_inferences() -> dict:
    records = list_inference_records()
    return {"items": records, "total": len(records)}


@app.delete("/api/inferences/{run_id}")
def remove_inference(run_id: str) -> dict:
    return {"item": delete_inference(run_id)}


@app.get("/api/training")
def get_training() -> dict:
    return training_status()


@app.post("/api/training/run", status_code=202)
def run_training() -> dict:
    _, started = start_training_job()
    return {
        "item": training_status(),
        "started": started,
    }


@app.get("/api/training/model")
def download_training_model() -> FileResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Ainda nao existe modelo treinado para download.")
    report = training_status()["latest_report"]
    filename = build_model_download_filename(report["trained_at"] if report else None)
    return FileResponse(MODEL_PATH, media_type="application/octet-stream", filename=filename)


@app.post("/api/inference")
def create_inference(image: UploadFile = File(...)) -> dict:
    return {"item": run_inference(image)}


@app.post("/api/inference/from-url")
def create_inference_from_url(payload: RemoteInferencePayload) -> dict:
    return {"item": run_inference_from_url(payload.image_url)}

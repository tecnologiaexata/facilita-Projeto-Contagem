from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import STORAGE_DIR
from app.services.monitoring import monitoring_snapshot
from app.services.annotation import (
    annotations_summary,
    build_annotation_package,
    delete_annotation,
    save_annotation,
)
from app.services.modeling import (
    MODEL_PATH,
    build_model_download_filename,
    delete_inference,
    run_inference,
    start_training_job,
    training_status,
)
from app.services.sam2 import Sam2PredictRequest, create_sam2_session, predict_for_session, sam2_status
from app.services.storage import (
    class_catalog,
    count_annotation_records,
    ensure_storage,
    load_annotation_record,
    list_annotation_records,
    list_inference_records,
)


ensure_storage()

app = FastAPI(title="Facilita Coffee Counter", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "classes": class_catalog(),
        "summary": annotations_summary(),
        "training": training_status(),
        "sam2": sam2_status(),
    }


@app.get("/api/annotations")
def get_annotations(
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1),
) -> dict:
    total = count_annotation_records()
    records = list_annotation_records(offset=offset, limit=limit)
    return {
        "items": records,
        "total": total,
        "offset": offset,
        "limit": limit if limit is not None else len(records),
    }


@app.post("/api/annotations")
def create_annotation(
    original_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    sample_id: str | None = Form(default=None),
) -> dict:
    record = save_annotation(original_image, mask_image, sample_id=sample_id)
    return {"item": record}


@app.get("/api/annotations/{sample_id}")
def get_annotation(sample_id: str) -> dict:
    record = load_annotation_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Anotacao nao encontrada.")
    return {"item": record}


@app.delete("/api/annotations/{sample_id}")
def remove_annotation(sample_id: str) -> dict:
    return {"item": delete_annotation(sample_id)}


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


@app.get("/api/sam2/status")
def get_sam2_status() -> dict:
    return sam2_status()


@app.post("/api/sam2/sessions")
def create_sam2_annotation_session(image: UploadFile = File(...)) -> dict:
    return {"item": create_sam2_session(image)}


@app.post("/api/sam2/sessions/{session_id}/predict")
def create_sam2_prediction(session_id: str, payload: Sam2PredictRequest) -> dict:
    return {"item": predict_for_session(session_id, payload)}


@app.get("/api/monitoring")
def get_monitoring() -> dict:
    return monitoring_snapshot()

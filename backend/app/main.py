from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_VERSION
from app.logging_utils import get_logger
from app.services.control_plane import (
    control_plane_status,
    start_control_plane_heartbeat,
    stop_control_plane_heartbeat,
)
from app.services.monitoring import monitoring_snapshot
from app.services.storage import class_catalog


logger = get_logger("facilita.worker.api")
app = FastAPI(title="Facilita Coffee Worker", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("API do worker iniciando. version=%s", APP_VERSION)
    start_control_plane_heartbeat()


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("API do worker encerrando.")
    stop_control_plane_heartbeat()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "role": "worker", "version": APP_VERSION}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "role": "worker",
        "version": APP_VERSION,
        "classes": class_catalog(),
        "worker": control_plane_status(),
    }


@app.get("/api/worker")
def worker_meta() -> dict:
    return {"item": control_plane_status()}


@app.get("/api/monitoring")
def monitoring() -> dict:
    return {"item": monitoring_snapshot()}

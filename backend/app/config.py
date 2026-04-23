import os
import platform


def default_worker_id() -> str:
    return platform.node().strip() or "facilita-coffee-worker"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


CLASS_MAP = {
    0: {"slug": "fundo", "label": "Fundo", "color": (229, 78, 78), "draw_order": 1, "annotatable": True},
    1: {"slug": "coffee", "label": "Coffee", "color": (196, 137, 78), "draw_order": 2, "annotatable": True},
    2: {"slug": "planta", "label": "Planta", "color": (104, 181, 106), "draw_order": 0, "annotatable": False},
}
CLASS_SLUG_ALIASES = {
    "background": "fundo",
    "bg": "fundo",
    "cafe": "coffee",
    "coffee": "coffee",
    "coffe": "coffee",
    "folhagem": "planta",
    "plant": "planta",
    "plants": "planta",
}
ANNOTATED_CLASS_IDS = tuple(
    class_id for class_id, meta in sorted(CLASS_MAP.items(), key=lambda item: item[0]) if meta.get("annotatable")
)
INFERRED_CLASS_ID = next(
    class_id for class_id, meta in sorted(CLASS_MAP.items(), key=lambda item: item[0]) if not meta.get("annotatable")
)
WORKER_CAPABILITIES = ("annotation_import", "training", "inference")


APP_VERSION = os.getenv("APP_VERSION", "0.3.0")
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN", "").strip()
BLOB_BASE_URL = os.getenv("BLOB_BASE_URL", "").strip()
BLOB_ACCESS = "private" if os.getenv("BLOB_ACCESS", "public").strip().lower() == "private" else "public"

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "").strip().rstrip("/")
WORKER_ID = os.getenv("WORKER_ID", "").strip() or default_worker_id()
WORKER_LABEL = os.getenv("WORKER_LABEL", "").strip() or "facilita-coffee-worker"
WORKER_PUBLIC_URL = os.getenv("WORKER_PUBLIC_URL", "").strip().rstrip("/")
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "").strip()
WORKER_DEFAULT_YOLO_DEVICE = os.getenv("WORKER_DEFAULT_YOLO_DEVICE", "").strip() or "0"
WORKER_DEFAULT_YOLO_MODEL = os.getenv("WORKER_DEFAULT_YOLO_MODEL", "").strip()
WORKER_MAX_CONCURRENT_JOBS = env_int("WORKER_MAX_CONCURRENT_JOBS", 1)
WORKER_HEARTBEAT_INTERVAL_SECONDS = max(5, env_int("WORKER_HEARTBEAT_INTERVAL_SECONDS", 15))
WORKER_HEARTBEAT_ENABLED = env_bool("WORKER_HEARTBEAT_ENABLED", True)
WORKER_JOB_POLL_ENABLED = env_bool("WORKER_JOB_POLL_ENABLED", True)
WORKER_JOB_POLL_INTERVAL_SECONDS = max(2, env_int("WORKER_JOB_POLL_INTERVAL_SECONDS", 5))
_job_stuck_after_seconds = env_int("WORKER_JOB_STUCK_AFTER_SECONDS", 300)
WORKER_JOB_STUCK_AFTER_SECONDS = 0 if _job_stuck_after_seconds <= 0 else max(30, _job_stuck_after_seconds)

REMOTE_FETCH_TIMEOUT_SECONDS = max(5, env_int("REMOTE_FETCH_TIMEOUT_SECONDS", 60))
REMOTE_FETCH_MAX_BYTES = max(1024 * 1024, env_int("REMOTE_FETCH_MAX_BYTES", 50 * 1024 * 1024))
REMOTE_FETCH_ALLOWED_HOSTS = tuple(
    host.strip().lower()
    for host in os.getenv("REMOTE_FETCH_ALLOWED_HOSTS", "").split(",")
    if host.strip()
)

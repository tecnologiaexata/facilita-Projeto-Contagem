import os
from pathlib import Path


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


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else BACKEND_DIR
STORAGE_DIR = PROJECT_DIR / "storage"
URL_ROOT_DIR = STORAGE_DIR.parent


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


ANNOTATIONS_DIR = STORAGE_DIR / "dataset_anotado"
ANNOTATION_IMAGES_DIR = ANNOTATIONS_DIR / "images"
ANNOTATION_IMAGE_PREVIEWS_DIR = ANNOTATIONS_DIR / "image_previews"
ANNOTATION_MASKS_DIR = ANNOTATIONS_DIR / "masks"
ANNOTATION_COLOR_MASKS_DIR = ANNOTATIONS_DIR / "colored_masks"
ANNOTATION_OVERLAYS_DIR = ANNOTATIONS_DIR / "overlays"
ANNOTATION_OVERLAY_PREVIEWS_DIR = ANNOTATIONS_DIR / "overlay_previews"
ANNOTATION_METADATA_DIR = ANNOTATIONS_DIR / "metadata"
ANNOTATION_TEXTS_DIR = ANNOTATIONS_DIR / "annotation_texts"

CVAT_DIR = STORAGE_DIR / "cvat"
DATASET_SPLIT_DIR = STORAGE_DIR / "dataset"
MODELS_DIR = STORAGE_DIR / "models"
TRAINING_DIR = STORAGE_DIR / "training"
INFERENCES_DIR = STORAGE_DIR / "inferences"


REQUIRED_DIRS = [
    STORAGE_DIR,
    ANNOTATION_IMAGES_DIR,
    ANNOTATION_IMAGE_PREVIEWS_DIR,
    ANNOTATION_MASKS_DIR,
    ANNOTATION_COLOR_MASKS_DIR,
    ANNOTATION_OVERLAYS_DIR,
    ANNOTATION_OVERLAY_PREVIEWS_DIR,
    ANNOTATION_METADATA_DIR,
    ANNOTATION_TEXTS_DIR,
    CVAT_DIR,
    DATASET_SPLIT_DIR,
    MODELS_DIR,
    TRAINING_DIR,
    INFERENCES_DIR,
]


APP_VERSION = os.getenv("APP_VERSION", "0.2.0")
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN", "").strip()
BLOB_BASE_URL = os.getenv("BLOB_BASE_URL", "").strip()
BLOB_ACCESS = "private" if os.getenv("BLOB_ACCESS", "public").strip().lower() == "private" else "public"

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "").strip().rstrip("/")
WORKER_ID = os.getenv("WORKER_ID", "").strip() or os.uname().nodename
WORKER_LABEL = os.getenv("WORKER_LABEL", "").strip() or "facilita-coffee-worker"
WORKER_PUBLIC_URL = os.getenv("WORKER_PUBLIC_URL", "").strip().rstrip("/")
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "").strip()
WORKER_MAX_CONCURRENT_JOBS = env_int("WORKER_MAX_CONCURRENT_JOBS", 1)
WORKER_HEARTBEAT_INTERVAL_SECONDS = max(5, env_int("WORKER_HEARTBEAT_INTERVAL_SECONDS", 15))
WORKER_HEARTBEAT_ENABLED = env_bool("WORKER_HEARTBEAT_ENABLED", True)
WORKER_JOB_POLL_ENABLED = env_bool("WORKER_JOB_POLL_ENABLED", True)
WORKER_JOB_POLL_INTERVAL_SECONDS = max(2, env_int("WORKER_JOB_POLL_INTERVAL_SECONDS", 5))

REMOTE_FETCH_TIMEOUT_SECONDS = max(5, env_int("REMOTE_FETCH_TIMEOUT_SECONDS", 60))
REMOTE_FETCH_MAX_BYTES = max(1024 * 1024, env_int("REMOTE_FETCH_MAX_BYTES", 50 * 1024 * 1024))
REMOTE_FETCH_ALLOWED_HOSTS = tuple(
    host.strip().lower()
    for host in os.getenv("REMOTE_FETCH_ALLOWED_HOSTS", "").split(",")
    if host.strip()
)

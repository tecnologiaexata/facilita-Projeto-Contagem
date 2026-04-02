import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import CLASS_MAP


DIR_MODE = 0o777


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(DIR_MODE)
    except PermissionError:
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_asset_id(prefix: str = "sample") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{timestamp}_{uuid4().hex[:12]}"


def normalize_request_id(request_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", request_id or "").strip("-").lower()
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return (cleaned or uuid4().hex[:12])[:80]


def make_stable_asset_id(prefix: str, request_id: str) -> str:
    return f"{prefix}_{normalize_request_id(request_id)}"


def build_split_map(records: list[dict]) -> dict[str, list[str]]:
    ordered_ids = [record["id"] for record in sorted(records, key=lambda item: item["created_at"])]
    total = len(ordered_ids)
    if total == 0:
        return {"train": [], "val": [], "test": []}
    if total == 1:
        return {"train": ordered_ids, "val": [], "test": []}
    if total == 2:
        return {"train": [ordered_ids[0]], "val": [ordered_ids[1]], "test": []}

    train_end = max(1, round(total * 0.7))
    val_size = max(1, round(total * 0.2))
    test_size = total - train_end - val_size
    if test_size <= 0:
        test_size = 1
        train_end = max(1, train_end - 1)

    val_end = min(total - test_size, train_end + val_size)
    return {
        "train": ordered_ids[:train_end],
        "val": ordered_ids[train_end:val_end],
        "test": ordered_ids[val_end:],
    }


def class_catalog() -> list[dict]:
    return [
        {"id": class_id, **meta}
        for class_id, meta in sorted(CLASS_MAP.items(), key=lambda item: item[0])
    ]

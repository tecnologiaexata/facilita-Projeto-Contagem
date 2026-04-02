import json
import mimetypes
from io import BytesIO
from urllib.parse import urlparse

from fastapi import HTTPException
from vercel.blob import BlobClient

from app.config import BLOB_ACCESS, BLOB_BASE_URL, BLOB_READ_WRITE_TOKEN


def blob_storage_enabled() -> bool:
    return bool(BLOB_READ_WRITE_TOKEN)


def ensure_blob_storage_ready() -> None:
    if blob_storage_enabled():
        return
    raise HTTPException(
        status_code=500,
        detail="BLOB_READ_WRITE_TOKEN nao configurado para uso do armazenamento definitivo.",
    )


def blob_access() -> str:
    return BLOB_ACCESS


def is_blob_reference(value: str | None) -> bool:
    if not value:
        return False

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc.endswith(".blob.vercel-storage.com"):
        return True

    if BLOB_BASE_URL and value.startswith(BLOB_BASE_URL):
        return True

    return False


def guess_filename_from_reference(reference: str | None, fallback: str) -> str:
    if not reference:
        return fallback

    parsed = urlparse(reference)
    candidate = parsed.path.rsplit("/", 1)[-1] if parsed.path else reference.rsplit("/", 1)[-1]
    return candidate or fallback


def _blob_client() -> BlobClient:
    ensure_blob_storage_ready()
    return BlobClient()


def _stream_to_bytes(stream) -> bytes:
    if stream is None:
        return b""
    if hasattr(stream, "read"):
        return stream.read()
    return b"".join(chunk for chunk in stream)


def download_blob_bytes(url_or_path: str, *, access: str | None = None) -> bytes:
    result = _blob_client().get(
        url_or_path,
        access=access or blob_access(),
    )
    if result is None or getattr(result, "status_code", 200) != 200 or getattr(result, "stream", None) is None:
        raise HTTPException(status_code=404, detail="Blob solicitado nao foi encontrado.")
    return _stream_to_bytes(result.stream)


def _normalize_uploaded_blob(uploaded, *, size: int | None = None, access: str | None = None) -> dict:
    uploaded_at = getattr(uploaded, "uploaded_at", None)
    return {
        "pathname": getattr(uploaded, "pathname", None),
        "content_type": getattr(uploaded, "content_type", None),
        "content_disposition": getattr(uploaded, "content_disposition", None),
        "url": getattr(uploaded, "url", None),
        "download_url": getattr(uploaded, "download_url", None),
        "cache_control": getattr(uploaded, "cache_control", None),
        "etag": getattr(uploaded, "etag", None),
        "size": getattr(uploaded, "size", size),
        "uploaded_at": uploaded_at.isoformat() if uploaded_at is not None else None,
        "access": access or blob_access(),
    }


def upload_blob_bytes(
    pathname: str,
    data: bytes,
    *,
    content_type: str | None = None,
    overwrite: bool = True,
    add_random_suffix: bool = False,
    access: str | None = None,
) -> dict:
    resolved_content_type = content_type or mimetypes.guess_type(pathname)[0] or "application/octet-stream"
    uploaded = _blob_client().put(
        pathname,
        data,
        access=access or blob_access(),
        content_type=resolved_content_type,
        overwrite=overwrite,
        add_random_suffix=add_random_suffix,
    )
    return _normalize_uploaded_blob(uploaded, size=len(data), access=access)


def upload_json_blob(pathname: str, payload: dict, *, access: str | None = None) -> dict:
    return upload_blob_bytes(
        pathname,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
        access=access,
    )


def load_image_from_blob_reference(reference: str, *, access: str | None = None) -> bytes:
    return download_blob_bytes(reference, access=access)

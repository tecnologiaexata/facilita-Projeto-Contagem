from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import HTTPException
from PIL import Image

from app.config import (
    REMOTE_FETCH_ALLOWED_HOSTS,
    REMOTE_FETCH_MAX_BYTES,
    REMOTE_FETCH_TIMEOUT_SECONDS,
)


CHUNK_SIZE = 1024 * 1024


def _validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="As URLs remotas precisam usar http ou https.")

    hostname = (parsed.hostname or "").lower()
    if REMOTE_FETCH_ALLOWED_HOSTS and hostname not in REMOTE_FETCH_ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail="A URL remota nao pertence a um host permitido para processamento.",
        )
    return url


def _download_remote_bytes(url: str) -> bytes:
    safe_url = _validate_remote_url(url)
    request = Request(safe_url, headers={"User-Agent": "facilita-coffee-worker/1.0"})
    total_read = 0
    chunks: list[bytes] = []

    try:
        with urlopen(request, timeout=REMOTE_FETCH_TIMEOUT_SECONDS) as response:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > REMOTE_FETCH_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="O arquivo remoto excede o tamanho maximo configurado para download.",
                    )
                chunks.append(chunk)
    except HTTPException:
        raise
    except HTTPError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Falha ao baixar arquivo remoto. HTTP {exc.code}.",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Falha de rede ao baixar arquivo remoto: {exc.reason}",
        ) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="O arquivo remoto retornou vazio.")
    return b"".join(chunks)


def filename_from_url(url: str, fallback: str) -> str:
    candidate = Path(urlparse(url).path).name.strip()
    return candidate or fallback


def fetch_remote_image(url: str, fallback_filename: str = "imagem.png") -> tuple[Image.Image, str]:
    payload = _download_remote_bytes(url)
    filename = filename_from_url(url, fallback_filename)
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Nao foi possivel interpretar a imagem remota.") from exc
    return image, filename


def fetch_remote_text(url: str, fallback_filename: str = "anotacao.txt") -> tuple[str, str]:
    payload = _download_remote_bytes(url)
    filename = filename_from_url(url, fallback_filename)
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = payload.decode(encoding)
        except UnicodeDecodeError:
            continue
        if content.strip():
            return content, filename

    raise HTTPException(status_code=400, detail="Nao foi possivel interpretar o TXT remoto.")

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.config import (
    REMOTE_FETCH_ALLOWED_HOSTS,
    REMOTE_FETCH_MAX_BYTES,
    REMOTE_FETCH_TIMEOUT_SECONDS,
)
from app.logging_utils import get_logger


CHUNK_SIZE = 1024 * 1024
logger = get_logger("facilita.worker.remote_assets")


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
    logger.info(
        "Baixando arquivo remoto: url=%s timeout=%s max_bytes=%s",
        safe_url,
        REMOTE_FETCH_TIMEOUT_SECONDS,
        REMOTE_FETCH_MAX_BYTES,
    )
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
                    logger.error(
                        "Arquivo remoto excedeu o limite configurado: url=%s bytes=%s limit=%s",
                        safe_url,
                        total_read,
                        REMOTE_FETCH_MAX_BYTES,
                    )
                    raise HTTPException(
                        status_code=413,
                        detail="O arquivo remoto excede o tamanho maximo configurado para download.",
                    )
                chunks.append(chunk)
    except HTTPException:
        raise
    except HTTPError as exc:
        logger.error("Falha HTTP ao baixar arquivo remoto: url=%s status=%s", safe_url, exc.code)
        raise HTTPException(
            status_code=400,
            detail=f"Falha ao baixar arquivo remoto. HTTP {exc.code}.",
        ) from exc
    except URLError as exc:
        logger.error("Falha de rede ao baixar arquivo remoto: url=%s reason=%s", safe_url, exc.reason)
        raise HTTPException(
            status_code=400,
            detail=f"Falha de rede ao baixar arquivo remoto: {exc.reason}",
        ) from exc

    if not chunks:
        logger.error("Arquivo remoto retornou vazio: url=%s", safe_url)
        raise HTTPException(status_code=400, detail="O arquivo remoto retornou vazio.")
    payload = b"".join(chunks)
    logger.info("Arquivo remoto baixado com sucesso: url=%s bytes=%s", safe_url, len(payload))
    return payload


def filename_from_url(url: str, fallback: str) -> str:
    candidate = Path(urlparse(url).path).name.strip()
    return candidate or fallback


def fetch_remote_image(url: str, fallback_filename: str = "imagem.png") -> tuple[Image.Image, str]:
    payload = _download_remote_bytes(url)
    filename = filename_from_url(url, fallback_filename)
    try:
        image = Image.open(BytesIO(payload))
        original_size = image.size
        image = ImageOps.exif_transpose(image)
        if image.size != original_size:
            logger.info(
                "Orientacao EXIF aplicada na imagem remota: url=%s filename=%s from=%s to=%s",
                url,
                filename,
                original_size,
                image.size,
            )
        image = image.convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Falha ao interpretar imagem remota: url=%s filename=%s", url, filename)
        raise HTTPException(status_code=400, detail="Nao foi possivel interpretar a imagem remota.") from exc
    logger.info("Imagem remota pronta para uso: url=%s filename=%s size=%sx%s", url, filename, image.width, image.height)
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
            logger.info("TXT remoto pronto para uso: url=%s filename=%s encoding=%s", url, filename, encoding)
            return content, filename

    logger.error("Falha ao interpretar TXT remoto: url=%s filename=%s", url, filename)
    raise HTTPException(status_code=400, detail="Nao foi possivel interpretar o TXT remoto.")

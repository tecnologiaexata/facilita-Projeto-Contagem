#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o worker backend do Facilita Coffee.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8050")))
    parser.add_argument("--reload", action="store_true", default=env_bool("RELOAD", False))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    return parser.parse_args()


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    args = parse_args()
    from app.logging_utils import configure_logging, get_logger

    configure_logging(args.log_level)
    logger = get_logger("facilita.worker.runner")
    logger.info(
        "Inicializando worker HTTP.",
        extra={},
    )
    logger.info(
        "Configuracao do servidor: host=%s port=%s reload=%s log_level=%s",
        args.host,
        args.port,
        args.reload,
        args.log_level,
    )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "uvicorn nao encontrado. Instale as dependencias com 'pip install -r backend/requirements.txt'."
        ) from exc

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        app_dir=str(BACKEND_DIR),
    )


if __name__ == "__main__":
    main()

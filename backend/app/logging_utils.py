import json
import logging


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level or "INFO").strip().upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def compact_json(payload: dict | None) -> str:
    if not payload:
        return "{}"
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        normalized = {key: str(value) for key, value in payload.items()}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)

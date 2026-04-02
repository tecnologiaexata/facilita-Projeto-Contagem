import json
import socket
import time
from threading import Event, Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from fastapi import HTTPException

from app.config import (
    APP_VERSION,
    BLOB_READ_WRITE_TOKEN,
    CONTROL_PLANE_URL,
    WORKER_CAPABILITIES,
    WORKER_HEARTBEAT_ENABLED,
    WORKER_HEARTBEAT_INTERVAL_SECONDS,
    WORKER_ID,
    WORKER_JOB_POLL_ENABLED,
    WORKER_JOB_POLL_INTERVAL_SECONDS,
    WORKER_LABEL,
    WORKER_MAX_CONCURRENT_JOBS,
    WORKER_PUBLIC_URL,
    WORKER_SHARED_TOKEN,
)
from app.logging_utils import compact_json, get_logger
from app.services.storage import now_iso
from app.services.worker_jobs import process_control_plane_job


logger = get_logger("facilita.worker.control_plane")
THREAD_LOCK = Lock()
STATE_LOCK = Lock()
STOP_EVENT = Event()
CONTROL_PLANE_THREAD: Thread | None = None
ACTIVE_JOB_IDS: set[str] = set()
ACTIVE_JOB_THREADS: dict[str, Thread] = {}
RUNTIME_STATE = {
    "enabled": False,
    "registered": False,
    "last_attempt_at": None,
    "last_success_at": None,
    "last_error": None,
    "last_job_poll_at": None,
    "last_job_claim_at": None,
    "last_job_complete_at": None,
    "last_job_id": None,
    "active_jobs": [],
    "active_job_count": 0,
}


def control_plane_enabled() -> bool:
    return bool(WORKER_HEARTBEAT_ENABLED and CONTROL_PLANE_URL and WORKER_PUBLIC_URL)


def _job_log_context(job: dict | None) -> dict:
    payload = job or {}
    return {
        "job_id": str(payload.get("id") or ""),
        "job_kind": str(payload.get("kind") or ""),
        "job_status": str(payload.get("status") or ""),
    }


def worker_registration_payload() -> dict:
    return {
        "worker_id": WORKER_ID,
        "label": WORKER_LABEL,
        "public_url": WORKER_PUBLIC_URL,
        "status": "online",
        "version": APP_VERSION,
        "capabilities": list(WORKER_CAPABILITIES),
        "max_concurrent_jobs": WORKER_MAX_CONCURRENT_JOBS,
        "metadata": {
            "hostname": socket.gethostname(),
            "blob_configured": bool(BLOB_READ_WRITE_TOKEN),
        },
    }


def _update_state(**changes) -> None:
    with STATE_LOCK:
        RUNTIME_STATE.update(changes)


def _set_active_job(job_id: str, is_active: bool) -> None:
    with STATE_LOCK:
        if is_active:
            ACTIVE_JOB_IDS.add(job_id)
        else:
            ACTIVE_JOB_IDS.discard(job_id)
            ACTIVE_JOB_THREADS.pop(job_id, None)
        RUNTIME_STATE["active_jobs"] = sorted(ACTIVE_JOB_IDS)
        RUNTIME_STATE["active_job_count"] = len(ACTIVE_JOB_IDS)


def _request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if WORKER_SHARED_TOKEN:
        headers["x-worker-token"] = WORKER_SHARED_TOKEN
    return headers


def _request_json(method: str, path: str, payload: dict | None = None) -> dict | None:
    endpoint = urljoin(f"{CONTROL_PLANE_URL}/", path.lstrip("/"))
    logger.info(
        "Enviando requisicao ao control plane: method=%s path=%s payload=%s",
        method.upper(),
        path,
        compact_json(payload),
    )
    request = Request(
        endpoint,
        method=method.upper(),
        headers=_request_headers(),
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8").strip()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        logger.error(
            "Falha HTTP ao comunicar com control plane: method=%s path=%s status=%s detail=%s",
            method.upper(),
            path,
            exc.code,
            detail or exc.reason,
        )
        raise RuntimeError(f"HTTP {exc.code} ao comunicar com control plane: {detail or exc.reason}") from exc
    except URLError as exc:
        logger.error(
            "Falha de rede ao comunicar com control plane: method=%s path=%s reason=%s",
            method.upper(),
            path,
            exc.reason,
        )
        raise RuntimeError(f"Falha de rede ao comunicar com control plane: {exc.reason}") from exc

    if not body:
        logger.info(
            "Control plane respondeu sem corpo: method=%s path=%s",
            method.upper(),
            path,
        )
        return None
    try:
        decoded = json.loads(body)
        logger.info(
            "Resposta recebida do control plane: method=%s path=%s body=%s",
            method.upper(),
            path,
            compact_json(decoded if isinstance(decoded, dict) else {"value": decoded}),
        )
        return decoded
    except json.JSONDecodeError:
        logger.warning(
            "Resposta nao JSON recebida do control plane: method=%s path=%s body=%s",
            method.upper(),
            path,
            body,
        )
        return {"raw": body}


def register_worker() -> dict | None:
    payload = worker_registration_payload()
    _update_state(last_attempt_at=now_iso())
    logger.info(
        "Registrando worker: worker_id=%s public_url=%s control_plane=%s",
        WORKER_ID,
        WORKER_PUBLIC_URL,
        CONTROL_PLANE_URL,
    )
    response = _request_json("POST", "/api/internal/backends/register", payload)
    _update_state(enabled=True, registered=True, last_success_at=now_iso(), last_error=None)
    logger.info("Worker registrado com sucesso.")
    return response


def send_heartbeat() -> dict | None:
    payload = worker_registration_payload()
    payload["last_seen_at"] = now_iso()
    _update_state(last_attempt_at=now_iso())
    logger.info("Enviando heartbeat do worker.")
    response = _request_json("POST", "/api/internal/backends/heartbeat", payload)
    _update_state(enabled=True, registered=True, last_success_at=now_iso(), last_error=None)
    logger.info("Heartbeat confirmado pelo control plane.")
    return response


def claim_next_job() -> dict | None:
    payload = {
        "worker_id": WORKER_ID,
        "public_url": WORKER_PUBLIC_URL,
    }
    _update_state(last_job_poll_at=now_iso())
    logger.info("Consultando fila por novo job.")
    response = _request_json("POST", "/api/internal/jobs/claim", payload)
    job = response.get("item") if isinstance(response, dict) else None
    if job:
        _update_state(
            last_job_claim_at=now_iso(),
            last_job_id=str(job.get("id") or ""),
            last_error=None,
        )
        logger.info(
            "Job recebido da fila: %s",
            compact_json(_job_log_context(job)),
        )
    else:
        logger.info("Nenhum job disponivel na fila neste ciclo.")
    return job


def load_job_context(job_id: str) -> dict | None:
    logger.info("Carregando contexto do job: job_id=%s", job_id)
    response = _request_json(
        "GET",
        f"/api/internal/jobs/{quote(job_id)}/context?worker_id={quote(WORKER_ID)}",
    )
    item = response.get("item") if isinstance(response, dict) else None
    logger.info(
        "Contexto carregado para job: job_id=%s has_context=%s",
        job_id,
        bool(item),
    )
    return item


def complete_job(job_id: str, response_payload: dict) -> dict | None:
    logger.info(
        "Reportando conclusao do job: job_id=%s payload=%s",
        job_id,
        compact_json({"keys": sorted((response_payload or {}).keys())}),
    )
    response = _request_json(
        "POST",
        f"/api/internal/jobs/{job_id}/complete",
        {
            "worker_id": WORKER_ID,
            "response_payload": response_payload,
        },
    )
    _update_state(last_job_complete_at=now_iso(), last_job_id=job_id, last_error=None)
    logger.info("Conclusao do job confirmada pelo control plane: job_id=%s", job_id)
    return response


def fail_job(job_id: str, error_payload: dict) -> dict | None:
    logger.error(
        "Reportando falha do job: job_id=%s payload=%s",
        job_id,
        compact_json(error_payload),
    )
    response = _request_json(
        "POST",
        f"/api/internal/jobs/{job_id}/fail",
        {
            "worker_id": WORKER_ID,
            "error_payload": error_payload,
        },
    )
    _update_state(last_job_complete_at=now_iso(), last_job_id=job_id, last_error=None)
    logger.info("Falha do job registrada no control plane: job_id=%s", job_id)
    return response


def _execute_job(job: dict) -> None:
    job_id = str(job.get("id") or "")
    if not job_id:
        logger.warning("Ignorando job sem id: payload=%s", compact_json(job))
        return

    job_context = _job_log_context(job)
    _set_active_job(job_id, True)
    logger.info("Iniciando execucao do job: %s", compact_json(job_context))
    try:
        context = load_job_context(job_id) or {}
        logger.info(
            "Contexto do job pronto: job_id=%s context_keys=%s",
            job_id,
            sorted(context.keys()),
        )
        result = process_control_plane_job(job, context)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        logger.error(
            "Job falhou com erro HTTP: job_id=%s kind=%s status_code=%s detail=%s",
            job_id,
            job.get("kind"),
            exc.status_code,
            detail,
        )
        try:
            fail_job(
                job_id,
                {
                    "error": detail,
                    "status_code": exc.status_code,
                    "job_kind": job.get("kind"),
                },
            )
        except Exception as report_exc:  # pragma: no cover - defensive
            logger.exception(
                "Falha ao reportar erro HTTP do job: job_id=%s reason=%s",
                job_id,
                report_exc,
            )
            _update_state(last_error=f"{detail} | Falha ao reportar erro do job: {report_exc}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Job falhou com excecao inesperada: job_id=%s kind=%s",
            job_id,
            job.get("kind"),
        )
        try:
            fail_job(
                job_id,
                {
                    "error": str(exc),
                    "job_kind": job.get("kind"),
                },
            )
        except Exception as report_exc:  # pragma: no cover - defensive
            logger.exception(
                "Falha ao reportar excecao inesperada do job: job_id=%s reason=%s",
                job_id,
                report_exc,
            )
            _update_state(last_error=f"{exc} | Falha ao reportar erro do job: {report_exc}")
    else:
        logger.info(
            "Job processado com sucesso localmente: job_id=%s result_keys=%s",
            job_id,
            sorted((result or {}).keys()),
        )
        try:
            complete_job(job_id, result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Falha ao reportar conclusao do job: job_id=%s reason=%s",
                job_id,
                exc,
            )
            _update_state(last_error=f"Falha ao reportar conclusao do job {job_id}: {exc}")
    finally:
        _set_active_job(job_id, False)
        logger.info("Execucao do job finalizada: job_id=%s", job_id)


def _try_start_jobs() -> None:
    if not WORKER_JOB_POLL_ENABLED:
        logger.info("Polling de jobs desativado por configuracao.")
        return

    while True:
        with STATE_LOCK:
            active_count = len(ACTIVE_JOB_IDS)
        if active_count >= WORKER_MAX_CONCURRENT_JOBS:
            logger.info(
                "Capacidade maxima atingida; nenhum novo job sera iniciado agora: active=%s max=%s",
                active_count,
                WORKER_MAX_CONCURRENT_JOBS,
            )
            return

        job = claim_next_job()
        if not job:
            return

        job_id = str(job.get("id") or "")
        if not job_id:
            logger.warning("Control plane retornou job sem id: payload=%s", compact_json(job))
            return

        worker_thread = Thread(
            target=_execute_job,
            args=(job,),
            daemon=True,
            name=f"control-plane-job-{job_id}",
        )
        with STATE_LOCK:
            ACTIVE_JOB_THREADS[job_id] = worker_thread
        worker_thread.start()
        logger.info("Thread iniciada para job: job_id=%s thread_name=%s", job_id, worker_thread.name)


def _control_plane_loop() -> None:
    registered = False
    last_heartbeat_monotonic = 0.0
    last_poll_monotonic = 0.0
    logger.info(
        "Loop do control plane iniciado: worker_id=%s control_plane=%s poll_enabled=%s",
        WORKER_ID,
        CONTROL_PLANE_URL,
        WORKER_JOB_POLL_ENABLED,
    )

    while not STOP_EVENT.is_set():
        now_monotonic = time.monotonic()

        try:
            if not registered or (now_monotonic - last_heartbeat_monotonic) >= WORKER_HEARTBEAT_INTERVAL_SECONDS:
                if not registered:
                    register_worker()
                    registered = True
                else:
                    send_heartbeat()
                last_heartbeat_monotonic = now_monotonic
        except Exception as exc:  # pragma: no cover - defensive
            _update_state(enabled=True, registered=False, last_error=str(exc))
            logger.exception("Falha durante registro/heartbeat do worker: %s", exc)
            registered = False

        if (
            registered
            and WORKER_JOB_POLL_ENABLED
            and (now_monotonic - last_poll_monotonic) >= WORKER_JOB_POLL_INTERVAL_SECONDS
        ):
            try:
                _try_start_jobs()
            except Exception as exc:  # pragma: no cover - defensive
                _update_state(last_error=str(exc))
                logger.exception("Falha durante polling/claim de jobs: %s", exc)
            last_poll_monotonic = now_monotonic

        STOP_EVENT.wait(1.0)

    logger.info("Loop do control plane encerrado.")


def start_control_plane_heartbeat() -> None:
    if not control_plane_enabled():
        logger.warning(
            "Heartbeat/polling nao iniciado porque CONTROL_PLANE_URL ou WORKER_PUBLIC_URL nao foram definidos."
        )
        _update_state(
            enabled=False,
            registered=False,
            last_error="Defina CONTROL_PLANE_URL e WORKER_PUBLIC_URL para ativar heartbeat e polling.",
        )
        return

    with THREAD_LOCK:
        global CONTROL_PLANE_THREAD

        if CONTROL_PLANE_THREAD is not None and CONTROL_PLANE_THREAD.is_alive():
            logger.info("Loop do control plane ja esta em execucao.")
            return

        STOP_EVENT.clear()
        _update_state(enabled=True, last_error=None)
        CONTROL_PLANE_THREAD = Thread(
            target=_control_plane_loop,
            daemon=True,
            name="control-plane-runtime",
        )
        CONTROL_PLANE_THREAD.start()
        logger.info("Thread do control plane iniciada: %s", CONTROL_PLANE_THREAD.name)


def stop_control_plane_heartbeat() -> None:
    STOP_EVENT.set()
    logger.info("Sinal de parada enviado para o control plane.")


def control_plane_status() -> dict:
    with STATE_LOCK:
        runtime_state = dict(RUNTIME_STATE)

    return {
        **runtime_state,
        "control_plane_url": CONTROL_PLANE_URL or None,
        "worker_id": WORKER_ID,
        "worker_label": WORKER_LABEL,
        "worker_public_url": WORKER_PUBLIC_URL or None,
        "heartbeat_interval_seconds": WORKER_HEARTBEAT_INTERVAL_SECONDS,
        "job_poll_enabled": WORKER_JOB_POLL_ENABLED,
        "job_poll_interval_seconds": WORKER_JOB_POLL_INTERVAL_SECONDS,
        "max_concurrent_jobs": WORKER_MAX_CONCURRENT_JOBS,
        "capabilities": list(WORKER_CAPABILITIES),
        "blob_configured": bool(BLOB_READ_WRITE_TOKEN),
        "version": APP_VERSION,
    }

import json
import socket
import time
from uuid import uuid4
from datetime import datetime, timezone
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
    WORKER_DEFAULT_YOLO_DEVICE,
    WORKER_HEARTBEAT_ENABLED,
    WORKER_HEARTBEAT_INTERVAL_SECONDS,
    WORKER_ID,
    WORKER_JOB_POLL_ENABLED,
    WORKER_JOB_POLL_INTERVAL_SECONDS,
    WORKER_JOB_STUCK_AFTER_SECONDS,
    WORKER_LABEL,
    WORKER_MAX_CONCURRENT_JOBS,
    WORKER_PUBLIC_URL,
    WORKER_SHARED_TOKEN,
)
from app.services.gpu_runtime import torch_runtime_info
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
ACTIVE_JOB_DETAILS: dict[str, dict] = {}
STUCK_WARNING_JOB_IDS: set[str] = set()
WORKER_RUNTIME_ID = str(uuid4())
WORKER_PROCESS_STARTED_AT = now_iso()
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
    "active_job_details": [],
    "has_stuck_jobs": False,
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
    with STATE_LOCK:
        current_jobs = len(ACTIVE_JOB_IDS)

    return {
        "worker_id": WORKER_ID,
        "label": WORKER_LABEL,
        "public_url": WORKER_PUBLIC_URL,
        "status": "online",
        "version": APP_VERSION,
        "capabilities": list(WORKER_CAPABILITIES),
        "current_jobs": current_jobs,
        "max_concurrent_jobs": WORKER_MAX_CONCURRENT_JOBS,
        "metadata": {
            "hostname": socket.gethostname(),
            "blob_configured": bool(BLOB_READ_WRITE_TOKEN),
            "runtime_id": WORKER_RUNTIME_ID,
            "process_started_at": WORKER_PROCESS_STARTED_AT,
        },
    }


def _update_state(**changes) -> None:
    with STATE_LOCK:
        RUNTIME_STATE.update(changes)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_since_iso(value: str | None, now: datetime | None = None) -> int | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    reference = now or _utc_now()
    return max(0, int((reference - parsed).total_seconds()))


def _snapshot_active_job_details_locked() -> list[dict]:
    now = _utc_now()
    details = []
    for job_id in sorted(ACTIVE_JOB_IDS):
        raw_detail = dict(ACTIVE_JOB_DETAILS.get(job_id) or {})
        started_at = raw_detail.get("started_at")
        last_progress_at = raw_detail.get("last_progress_at") or started_at
        runtime_seconds = _seconds_since_iso(started_at, now)
        seconds_since_progress = _seconds_since_iso(last_progress_at, now)
        is_stuck = bool(
            WORKER_JOB_STUCK_AFTER_SECONDS
            and seconds_since_progress is not None
            and seconds_since_progress >= WORKER_JOB_STUCK_AFTER_SECONDS
        )
        details.append(
            {
                **raw_detail,
                "job_id": job_id,
                "runtime_seconds": runtime_seconds,
                "seconds_since_progress": seconds_since_progress,
                "is_stuck": is_stuck,
                "stuck_after_seconds": WORKER_JOB_STUCK_AFTER_SECONDS or None,
            }
        )
    return details


def _refresh_active_job_state_locked() -> None:
    active_job_details = _snapshot_active_job_details_locked()
    RUNTIME_STATE["active_jobs"] = sorted(ACTIVE_JOB_IDS)
    RUNTIME_STATE["active_job_count"] = len(ACTIVE_JOB_IDS)
    RUNTIME_STATE["active_job_details"] = active_job_details
    RUNTIME_STATE["has_stuck_jobs"] = any(job.get("is_stuck") for job in active_job_details)


def _register_active_job(job: dict, *, thread_name: str | None = None) -> None:
    job_id = str(job.get("id") or "")
    if not job_id:
        return

    now = now_iso()
    with STATE_LOCK:
        ACTIVE_JOB_IDS.add(job_id)
        ACTIVE_JOB_DETAILS[job_id] = {
            "job_id": job_id,
            "kind": str(job.get("kind") or ""),
            "status": "running",
            "stage": "claimed",
            "stage_label": "Job recebido pelo worker.",
            "started_at": now,
            "last_progress_at": now,
            "thread_name": thread_name,
            "details": {},
        }
        _refresh_active_job_state_locked()


def _mark_job_progress(
    job_id: str,
    stage: str,
    detail: str | None = None,
    *,
    details: dict | None = None,
    touch: bool = True,
) -> None:
    if not job_id:
        return

    with STATE_LOCK:
        current = ACTIVE_JOB_DETAILS.get(job_id)
        if not current:
            return

        current["stage"] = stage
        if detail is not None:
            current["stage_label"] = detail
        if details:
            current["details"] = {
                **(current.get("details") or {}),
                **details,
            }
        if touch:
            current["last_progress_at"] = now_iso()
        _refresh_active_job_state_locked()


def _clear_active_job(job_id: str) -> None:
    with STATE_LOCK:
        ACTIVE_JOB_IDS.discard(job_id)
        ACTIVE_JOB_THREADS.pop(job_id, None)
        ACTIVE_JOB_DETAILS.pop(job_id, None)
        STUCK_WARNING_JOB_IDS.discard(job_id)
        _refresh_active_job_state_locked()


def _warn_for_stuck_jobs_once() -> None:
    if not WORKER_JOB_STUCK_AFTER_SECONDS:
        return

    stuck_jobs = []
    with STATE_LOCK:
        for job in _snapshot_active_job_details_locked():
            job_id = str(job.get("job_id") or "")
            if not job.get("is_stuck") or job_id in STUCK_WARNING_JOB_IDS:
                continue
            STUCK_WARNING_JOB_IDS.add(job_id)
            stuck_jobs.append(job)
        _refresh_active_job_state_locked()

    for job in stuck_jobs:
        logger.warning(
            "Job sem progresso ha tempo demais: job_id=%s kind=%s stage=%s seconds_since_progress=%s stuck_after=%s",
            job.get("job_id"),
            job.get("kind"),
            job.get("stage"),
            job.get("seconds_since_progress"),
            job.get("stuck_after_seconds"),
        )


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


def _execute_job(job: dict, thread_name: str | None = None) -> None:
    job_id = str(job.get("id") or "")
    if not job_id:
        logger.warning("Ignorando job sem id: payload=%s", compact_json(job))
        return

    job_context = _job_log_context(job)
    _register_active_job(job, thread_name=thread_name)
    logger.info("Iniciando execucao do job: %s", compact_json(job_context))
    try:
        _mark_job_progress(job_id, "loading_context", "Carregando contexto do job.")
        context = load_job_context(job_id) or {}
        context_keys = sorted(context.keys())
        logger.info(
            "Contexto do job pronto: job_id=%s context_keys=%s",
            job_id,
            context_keys,
        )
        _mark_job_progress(
            job_id,
            "processing",
            "Contexto carregado; iniciando processamento.",
            details={"context_keys": context_keys},
        )

        def report_progress(stage, detail=None, *, details=None, touch=True):
            _mark_job_progress(
                job_id,
                stage,
                detail,
                details=details,
                touch=touch,
            )

        result = process_control_plane_job(
            job,
            context,
            report_progress=report_progress,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        _mark_job_progress(
            job_id,
            "failed",
            "Job falhou durante o processamento.",
            details={
                "error": detail,
                "status_code": exc.status_code,
            },
            touch=False,
        )
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
        _mark_job_progress(
            job_id,
            "failed",
            "Job falhou com excecao inesperada.",
            details={"error": str(exc)},
            touch=False,
        )
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
        _mark_job_progress(
            job_id,
            "reporting_success",
            "Processamento concluido; reportando resultado ao control plane.",
            details={"result_keys": sorted((result or {}).keys())},
        )
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
        _clear_active_job(job_id)
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
            args=(job, f"control-plane-job-{job_id}"),
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

        _warn_for_stuck_jobs_once()

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
        _refresh_active_job_state_locked()
        runtime_state = dict(RUNTIME_STATE)
        runtime_state["active_job_details"] = [dict(job) for job in RUNTIME_STATE.get("active_job_details", [])]

    return {
        **runtime_state,
        "control_plane_url": CONTROL_PLANE_URL or None,
        "worker_id": WORKER_ID,
        "worker_label": WORKER_LABEL,
        "worker_public_url": WORKER_PUBLIC_URL or None,
        "heartbeat_interval_seconds": WORKER_HEARTBEAT_INTERVAL_SECONDS,
        "job_poll_enabled": WORKER_JOB_POLL_ENABLED,
        "job_poll_interval_seconds": WORKER_JOB_POLL_INTERVAL_SECONDS,
        "job_stuck_after_seconds": WORKER_JOB_STUCK_AFTER_SECONDS or None,
        "max_concurrent_jobs": WORKER_MAX_CONCURRENT_JOBS,
        "capabilities": list(WORKER_CAPABILITIES),
        "blob_configured": bool(BLOB_READ_WRITE_TOKEN),
        "compute_policy": {
            "default_yolo_device": WORKER_DEFAULT_YOLO_DEVICE,
            "training_requires_gpu": True,
            "inference_requires_gpu": True,
        },
        "torch_runtime": torch_runtime_info(),
        "version": APP_VERSION,
        "current_job": runtime_state["active_job_details"][0] if runtime_state.get("active_job_details") else None,
    }

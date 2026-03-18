import os
import socket
import subprocess
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, get_ident
from uuid import uuid4

import psutil

from app.config import SAM2_DEVICE
from app.services.storage import now_iso


BYTES_IN_MB = 1024 * 1024


@dataclass
class TaskRecord:
    id: str
    kind: str
    label: str
    status: str
    created_at: str
    started_perf: float
    started_iso: str
    phase: str
    pid: int
    thread_id: int
    start_cpu_time_seconds: float
    start_rss_bytes: int
    metadata: dict = field(default_factory=dict)
    finished_at: str | None = None
    finished_perf: float | None = None
    error: str | None = None

    def update(self, phase: str | None = None, metadata: dict | None = None) -> None:
        if phase:
            self.phase = phase
        if metadata:
            self.metadata.update(metadata)


class TaskHandle:
    def __init__(self, registry: "MonitoringRegistry", task_id: str) -> None:
        self.registry = registry
        self.task_id = task_id

    def update(self, phase: str | None = None, metadata: dict | None = None) -> None:
        self.registry.update_task(self.task_id, phase=phase, metadata=metadata)


class MonitoringRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active_tasks: dict[str, TaskRecord] = {}
        self._recent_tasks: deque[dict] = deque(maxlen=20)
        self._process = psutil.Process(os.getpid())
        self._hostname = socket.gethostname()
        self._cpu_count = psutil.cpu_count() or 1
        self._gpu_monitoring_enabled = not SAM2_DEVICE.lower().startswith("cpu")
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

    def _current_cpu_time(self) -> float:
        cpu_times = self._process.cpu_times()
        return float(cpu_times.user + cpu_times.system)

    def _current_rss(self) -> int:
        return int(self._process.memory_info().rss)

    def start_task(self, kind: str, label: str, metadata: dict | None = None) -> TaskHandle:
        task_id = f"task_{uuid4().hex[:12]}"
        record = TaskRecord(
            id=task_id,
            kind=kind,
            label=label,
            status="running",
            created_at=now_iso(),
            started_perf=time.perf_counter(),
            started_iso=now_iso(),
            phase="Iniciando",
            pid=self._process.pid,
            thread_id=get_ident(),
            start_cpu_time_seconds=self._current_cpu_time(),
            start_rss_bytes=self._current_rss(),
            metadata=metadata or {},
        )
        with self._lock:
            self._active_tasks[task_id] = record
        return TaskHandle(self, task_id)

    def update_task(self, task_id: str, phase: str | None = None, metadata: dict | None = None) -> None:
        with self._lock:
            record = self._active_tasks.get(task_id)
            if record is None:
                return
            record.update(phase=phase, metadata=metadata)

    def finish_task(self, task_id: str, status: str = "completed", error: str | None = None) -> None:
        with self._lock:
            record = self._active_tasks.pop(task_id, None)
        if record is None:
            return
        record.status = status
        record.error = error
        record.finished_at = now_iso()
        record.finished_perf = time.perf_counter()
        self._recent_tasks.appendleft(self._serialize_finished_task(record))

    def _serialize_active_task(self, record: TaskRecord, current_cpu_time: float, current_rss: int) -> dict:
        elapsed = max(0.0, time.perf_counter() - record.started_perf)
        cpu_time_delta = max(0.0, current_cpu_time - record.start_cpu_time_seconds)
        estimated_cpu_percent = (
            round((cpu_time_delta / elapsed) * 100 / self._cpu_count, 2)
            if elapsed > 0 and self._cpu_count > 0
            else 0.0
        )
        return {
            "id": record.id,
            "kind": record.kind,
            "label": record.label,
            "status": record.status,
            "phase": record.phase,
            "created_at": record.created_at,
            "started_at": record.started_iso,
            "elapsed_seconds": round(elapsed, 2),
            "cpu_time_seconds": round(cpu_time_delta, 2),
            "estimated_cpu_percent": estimated_cpu_percent,
            "process_rss_mb": round(current_rss / BYTES_IN_MB, 2),
            "memory_delta_mb": round((current_rss - record.start_rss_bytes) / BYTES_IN_MB, 2),
            "pid": record.pid,
            "thread_id": record.thread_id,
            "metadata": record.metadata,
        }

    def _serialize_finished_task(self, record: TaskRecord) -> dict:
        finished_perf = record.finished_perf or time.perf_counter()
        elapsed = max(0.0, finished_perf - record.started_perf)
        final_cpu_time = self._current_cpu_time()
        final_rss = self._current_rss()
        cpu_time_delta = max(0.0, final_cpu_time - record.start_cpu_time_seconds)
        return {
            "id": record.id,
            "kind": record.kind,
            "label": record.label,
            "status": record.status,
            "phase": record.phase,
            "created_at": record.created_at,
            "started_at": record.started_iso,
            "finished_at": record.finished_at,
            "elapsed_seconds": round(elapsed, 2),
            "cpu_time_seconds": round(cpu_time_delta, 2),
            "process_rss_mb": round(final_rss / BYTES_IN_MB, 2),
            "memory_delta_mb": round((final_rss - record.start_rss_bytes) / BYTES_IN_MB, 2),
            "metadata": record.metadata,
            "error": record.error,
        }

    def _gpu_snapshot(self) -> dict:
        if not self._gpu_monitoring_enabled:
            return {
                "available": False,
                "reason": "Monitoramento de GPU desativado quando o ambiente esta configurado para CPU.",
            }
        try:
            response = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
        except FileNotFoundError:
            return {"available": False, "reason": "nvidia-smi nao encontrado."}
        except subprocess.SubprocessError as exc:
            return {"available": False, "reason": f"Falha ao consultar GPU: {exc}"}

        gpus = []
        total_util = 0.0
        total_memory_used = 0.0
        total_memory_total = 0.0
        lines = [line.strip() for line in response.stdout.splitlines() if line.strip()]
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            util = float(parts[2])
            memory_used = float(parts[3])
            memory_total = float(parts[4])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_percent": round(util, 2),
                    "memory_used_mb": round(memory_used, 2),
                    "memory_total_mb": round(memory_total, 2),
                    "temperature_c": round(float(parts[5]), 2),
                }
            )
            total_util += util
            total_memory_used += memory_used
            total_memory_total += memory_total

        if not gpus:
            return {"available": False, "reason": "Nenhuma GPU retornada pelo nvidia-smi."}

        return {
            "available": True,
            "count": len(gpus),
            "total_utilization_percent": round(total_util / len(gpus), 2),
            "memory_used_mb": round(total_memory_used, 2),
            "memory_total_mb": round(total_memory_total, 2),
            "memory_percent": round((total_memory_used / total_memory_total) * 100, 2)
            if total_memory_total
            else 0.0,
            "devices": gpus,
        }

    def snapshot(self) -> dict:
        current_cpu_time = self._current_cpu_time()
        current_rss = self._current_rss()
        with self._lock:
            active_tasks = [
                self._serialize_active_task(record, current_cpu_time, current_rss)
                for record in self._active_tasks.values()
            ]
            recent_tasks = list(self._recent_tasks)

        active_tasks.sort(key=lambda item: item["started_at"])
        vm = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        process_cpu_percent = self._process.cpu_percent(interval=None)
        return {
            "generated_at": now_iso(),
            "server": {
                "hostname": self._hostname,
                "pid": self._process.pid,
                "uptime_seconds": round(time.time() - self._process.create_time(), 2),
                "cpu": {
                    "total_percent": round(cpu_percent, 2),
                    "core_count": self._cpu_count,
                    "process_percent": round(process_cpu_percent, 2),
                },
                "memory": {
                    "total_mb": round(vm.total / BYTES_IN_MB, 2),
                    "used_mb": round(vm.used / BYTES_IN_MB, 2),
                    "available_mb": round(vm.available / BYTES_IN_MB, 2),
                    "percent": round(vm.percent, 2),
                    "process_rss_mb": round(current_rss / BYTES_IN_MB, 2),
                },
                "gpu": self._gpu_snapshot(),
            },
            "tasks": {
                "active_count": len(active_tasks),
                "active": active_tasks,
                "recent": recent_tasks,
            },
        }

    def active_tasks(self, kind: str | None = None) -> list[dict]:
        current_cpu_time = self._current_cpu_time()
        current_rss = self._current_rss()
        with self._lock:
            items = [
                self._serialize_active_task(record, current_cpu_time, current_rss)
                for record in self._active_tasks.values()
                if kind is None or record.kind == kind
            ]
        items.sort(key=lambda item: item["started_at"])
        return items

    def recent_tasks(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = [
                dict(item)
                for item in self._recent_tasks
                if kind is None or item["kind"] == kind
            ]
        return items


MONITORING = MonitoringRegistry()


@contextmanager
def tracked_task(kind: str, label: str, metadata: dict | None = None):
    handle = MONITORING.start_task(kind=kind, label=label, metadata=metadata)
    try:
        yield handle
    except Exception as exc:
        MONITORING.finish_task(handle.task_id, status="failed", error=str(exc))
        raise
    else:
        MONITORING.finish_task(handle.task_id, status="completed")


def monitoring_snapshot() -> dict:
    return MONITORING.snapshot()


def list_active_tasks(kind: str | None = None) -> list[dict]:
    return MONITORING.active_tasks(kind=kind)


def list_recent_tasks(kind: str | None = None) -> list[dict]:
    return MONITORING.recent_tasks(kind=kind)

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
LOG_DIR="${REPO_ROOT}/logs"
RUN_DIR="${REPO_ROOT}/run"
PID_FILE="${RUN_DIR}/worker.pid"
REQUIREMENTS_FILE="${REPO_ROOT}/backend/requirements.txt"
REQUIREMENTS_STAMP="${VENV_DIR}/.requirements.sha256"
ENV_FILE="${REPO_ROOT}/.env"

log() {
  printf '[facilita-worker] %s\n' "$*"
}

fail() {
  printf '[facilita-worker] erro: %s\n' "$*" >&2
  exit 1
}

ensure_dirs() {
  mkdir -p "${LOG_DIR}" "${RUN_DIR}"
}

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi
}

worker_host() {
  printf '%s' "${HOST:-0.0.0.0}"
}

worker_port() {
  printf '%s' "${PORT:-8050}"
}

worker_log_file() {
  printf '%s' "${LOG_DIR}/worker.log"
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi

  rm -f "${PID_FILE}"
  return 1
}

requirements_hash() {
  local requirements_sha pytorch_mode pytorch_index
  requirements_sha="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
  pytorch_mode="${PYTORCH_INSTALL_MODE:-cuda}"
  pytorch_index="$(pytorch_index_url)"
  printf '%s' "${requirements_sha}|${pytorch_mode}|${pytorch_index}" | sha256sum | awk '{print $1}'
}

ensure_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "criando ambiente virtual em ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" || fail "nao foi possivel criar .venv. Instale python3-venv."
  fi
}

has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1
}

gpu_names() {
  if ! has_nvidia_gpu; then
    return 0
  fi

  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
}

has_blackwell_gpu() {
  local names
  names="$(gpu_names)"
  [[ -n "${names}" ]] || return 1
  grep -Eiq 'B200|B100|GB200|GB300|Blackwell|RTX PRO 6000 Blackwell|RTX 50' <<<"${names}"
}

index_url_supports_blackwell() {
  local index_url="${1:-}"
  [[ -n "${index_url}" ]] || return 1
  if [[ "${index_url}" =~ /cu([0-9]{3})/?$ ]]; then
    local version="${BASH_REMATCH[1]}"
    [[ "${version}" -ge 128 ]]
    return
  fi
  return 1
}

default_pytorch_index_url() {
  if has_blackwell_gpu; then
    printf '%s' "https://download.pytorch.org/whl/cu128"
  else
    printf '%s' "https://download.pytorch.org/whl/cu124"
  fi
}

pytorch_install_mode() {
  printf '%s' "${PYTORCH_INSTALL_MODE:-cuda}"
}

pytorch_index_url() {
  local configured="${PYTORCH_INDEX_URL:-}"
  local fallback
  fallback="$(default_pytorch_index_url)"

  if [[ -z "${configured}" ]]; then
    printf '%s' "${fallback}"
    return
  fi

  if has_blackwell_gpu && ! index_url_supports_blackwell "${configured}"; then
    log "GPU Blackwell detectada; substituindo PYTORCH_INDEX_URL=${configured} por ${fallback}"
    printf '%s' "${fallback}"
    return
  fi

  printf '%s' "${configured}"
}

ensure_pytorch_runtime() {
  local mode index_url
  mode="$(pytorch_install_mode)"
  index_url="$(pytorch_index_url)"

  case "${mode}" in
    skip)
      log "instalacao do PyTorch ignorada por PYTORCH_INSTALL_MODE=skip"
      return
      ;;
    cpu)
      log "instalando PyTorch CPU-only"
      "${VENV_DIR}/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      return
      ;;
    cuda)
      log "instalando PyTorch com CUDA a partir de ${index_url}"
      "${VENV_DIR}/bin/pip" install torch torchvision torchaudio --index-url "${index_url}"
      return
      ;;
    auto)
      if has_nvidia_gpu; then
        log "GPU detectada; instalando PyTorch com CUDA a partir de ${index_url}"
        "${VENV_DIR}/bin/pip" install torch torchvision torchaudio --index-url "${index_url}"
      else
        log "nenhuma GPU detectada via nvidia-smi; mantendo modo automatico sem instalar wheel CUDA dedicada"
      fi
      return
      ;;
    *)
      fail "PYTORCH_INSTALL_MODE invalido: ${mode}. Use auto, cuda, cpu ou skip."
      ;;
  esac
}

verify_torch_cuda() {
  local mode
  mode="$(pytorch_install_mode)"

  "${VENV_DIR}/bin/python" - <<'PY'
import json
import sys

try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"Falha ao importar torch: {exc}"}, ensure_ascii=False))
    raise SystemExit(1)

payload = {
    "ok": True,
    "torch_version": getattr(torch, "__version__", None),
    "cuda_build_version": getattr(getattr(torch, "version", None), "cuda", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}
print(json.dumps(payload, ensure_ascii=False))
PY

  if [[ "${mode}" == "cuda" ]]; then
    "${VENV_DIR}/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch foi configurado para CUDA, mas torch.cuda.is_available() retornou false.")
PY
  fi
}

ensure_dependencies() {
  ensure_venv

  local current_hash=""
  current_hash="$(requirements_hash)"
  local installed_hash=""
  if [[ -f "${REQUIREMENTS_STAMP}" ]]; then
    installed_hash="$(cat "${REQUIREMENTS_STAMP}")"
  fi

  if [[ "${current_hash}" == "${installed_hash}" ]]; then
    log "dependencias ja estao sincronizadas"
    return
  fi

  log "instalando dependencias Python"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  ensure_pytorch_runtime
  "${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS_FILE}"
  verify_torch_cuda
  printf '%s' "${current_hash}" > "${REQUIREMENTS_STAMP}"
}

wait_for_health() {
  load_env

  local host port url
  host="$(worker_host)"
  port="$(worker_port)"
  url="http://127.0.0.1:${port}/api/health"

  for _ in $(seq 1 30); do
    if curl --silent --fail "${url}" >/dev/null 2>&1; then
      log "worker respondeu em ${url}"
      return 0
    fi
    sleep 1
  done

  fail "worker nao respondeu em ${url} dentro do tempo esperado"
}

start_worker() {
  ensure_dirs
  load_env

  [[ -f "${ENV_FILE}" ]] || fail "arquivo .env nao encontrado em ${REPO_ROOT}"

  if is_running; then
    log "worker ja esta rodando com pid $(cat "${PID_FILE}")"
    return 0
  fi

  ensure_dependencies

  local logfile host port
  logfile="$(worker_log_file)"
  host="$(worker_host)"
  port="$(worker_port)"

  log "subindo worker em ${host}:${port}"
  (
    cd "${REPO_ROOT}"
    nohup "${VENV_DIR}/bin/python" "${REPO_ROOT}/run_worker.py" --host "${host}" --port "${port}" >>"${logfile}" 2>&1 &
    echo $! > "${PID_FILE}"
  )

  wait_for_health
  log "worker iniciado com pid $(cat "${PID_FILE}")"
}

stop_worker() {
  if ! is_running; then
    log "worker nao esta rodando"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  log "parando worker pid ${pid}"
  kill "${pid}" >/dev/null 2>&1 || true

  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${PID_FILE}"
      log "worker parado"
      return 0
    fi
    sleep 1
  done

  log "encerrando worker com kill -9"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
}

status_worker() {
  load_env

  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    log "worker rodando com pid ${pid}"
    printf 'host=%s port=%s log=%s\n' "$(worker_host)" "$(worker_port)" "$(worker_log_file)"
    return 0
  fi

  log "worker parado"
}

tail_logs() {
  ensure_dirs
  touch "$(worker_log_file)"
  tail -n 100 -f "$(worker_log_file)"
}

health_worker() {
  load_env
  curl --silent --fail "http://127.0.0.1:$(worker_port)/api/health"
  printf '\n'
}

restart_worker() {
  stop_worker
  start_worker
}

usage() {
  cat <<'EOF'
Uso:
  bash scripts/workerctl.sh bootstrap
  bash scripts/workerctl.sh start
  bash scripts/workerctl.sh stop
  bash scripts/workerctl.sh restart
  bash scripts/workerctl.sh status
  bash scripts/workerctl.sh health
  bash scripts/workerctl.sh logs
  bash scripts/workerctl.sh verify-gpu

Comportamento:
  - le o .env na raiz do projeto
  - cria .venv automaticamente se necessario
  - instala dependencias se backend/requirements.txt mudou
  - sobe run_worker.py em background e guarda pid em run/worker.pid
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    bootstrap)
      ensure_dirs
      load_env
      ensure_dependencies
      ;;
    start)
      start_worker
      ;;
    stop)
      stop_worker
      ;;
    restart)
      restart_worker
      ;;
    status)
      status_worker
      ;;
    health)
      health_worker
      ;;
    verify-gpu)
      load_env
      ensure_venv
      verify_torch_cuda
      ;;
    logs)
      tail_logs
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"

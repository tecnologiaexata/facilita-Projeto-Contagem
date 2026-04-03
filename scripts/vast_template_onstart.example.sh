#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

PROJECT_DIR="/workspace/Facilita-Projeto-Coffee"
REPO_URL="https://github.com/tecnologiaexata/Facilita-Projeto-Coffee.git"

# Preencha estes valores antes de colar no template da Vast.ai
CONTROL_PLANE_URL="https://facilita-projeto-coffee-frontend.vercel.app"
WORKER_SHARED_TOKEN="change-me"
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_3a971KbocryDE9JJ_EgmEMR41fh8gIOZX6t0H4PWv6I3Wn8"
BLOB_ACCESS="public"

install_base_packages() {
  apt-get update
  apt-get install -y git curl ca-certificates python3 python3-pip python3-venv
}

ensure_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    install_base_packages
  fi
}

ensure_runtime() {
  ensure_command git
  ensure_command curl
  ensure_command python3

  if ! python3 -m venv --help >/dev/null 2>&1; then
    install_base_packages
  fi
}

resolve_public_ip() {
  local public_ip=""
  public_ip="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
  if [[ -n "${public_ip}" ]]; then
    printf '%s' "${public_ip}"
    return 0
  fi

  public_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "${public_ip}" ]]; then
    printf '%s' "${public_ip}"
    return 0
  fi

  printf '%s' "127.0.0.1"
}

sync_repo() {
  mkdir -p /workspace

  if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
    rm -rf "${PROJECT_DIR}"
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    return 0
  fi

  cd "${PROJECT_DIR}"
  git pull --ff-only
}

write_env_file() {
  local public_ip worker_id worker_label

  public_ip="$(resolve_public_ip)"
  worker_id="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-')"
  worker_label="facilita-coffee-worker"

  cd "${PROJECT_DIR}"
  rm -f .env

  cat > .env <<EOF
# HTTP server
HOST=0.0.0.0
PORT=8050
RELOAD=false
LOG_LEVEL=info

# Worker identity
APP_VERSION=0.3.0
WORKER_ID=${worker_id:-vast-worker}
WORKER_LABEL=${worker_label}
WORKER_PUBLIC_URL=http://${public_ip}:8050
WORKER_MAX_CONCURRENT_JOBS=1

# Control plane on Vercel
CONTROL_PLANE_URL=${CONTROL_PLANE_URL}
WORKER_SHARED_TOKEN=${WORKER_SHARED_TOKEN}
WORKER_HEARTBEAT_ENABLED=true
WORKER_HEARTBEAT_INTERVAL_SECONDS=15
WORKER_JOB_POLL_ENABLED=true
WORKER_JOB_POLL_INTERVAL_SECONDS=5
WORKER_JOB_STUCK_AFTER_SECONDS=300

# Remote asset download
REMOTE_FETCH_TIMEOUT_SECONDS=6000
REMOTE_FETCH_MAX_BYTES=52428800
REMOTE_FETCH_ALLOWED_HOSTS=

# Blob storage
BLOB_READ_WRITE_TOKEN=${BLOB_READ_WRITE_TOKEN}
BLOB_BASE_URL=
BLOB_ACCESS=${BLOB_ACCESS}
EOF
}

main() {
  ensure_runtime
  sync_repo
  write_env_file

  cd "${PROJECT_DIR}"
  bash onstart.sh
}

main "$@"

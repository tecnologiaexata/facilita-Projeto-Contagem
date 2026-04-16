from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def torch_runtime_info() -> dict:
    info = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_build_version": None,
        "cuda_available": False,
        "cuda_check_error": None,
        "device_count": 0,
        "device_names": [],
        "torch_import_error": None,
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on worker environment
        info["torch_import_error"] = str(exc)
        return info

    info["torch_installed"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["cuda_build_version"] = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on driver/runtime
        info["cuda_check_error"] = str(exc)
        return info

    info["cuda_available"] = cuda_available
    if not cuda_available:
        return info

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - defensive
        info["cuda_check_error"] = str(exc)
        return info

    info["device_count"] = device_count
    names = []
    for index in range(device_count):
        try:
            names.append(str(torch.cuda.get_device_name(index)))
        except Exception as exc:  # pragma: no cover - defensive
            names.append(f"GPU {index} indisponivel ({exc})")
    info["device_names"] = names
    return info


def normalize_requested_device(requested_device, *, fallback_device: str = "0") -> str:
    value = str(requested_device).strip() if requested_device is not None else ""
    return value or fallback_device


def require_gpu_device(requested_device, *, operation: str, fallback_device: str = "0") -> str:
    device = normalize_requested_device(requested_device, fallback_device=fallback_device)
    if device.lower() == "cpu":
        raise RuntimeError(f"{operation} exige GPU CUDA. O device configurado foi 'cpu'.")

    runtime = torch_runtime_info()
    if not runtime["torch_installed"]:
        raise RuntimeError(
            f"{operation} exige GPU CUDA, mas o PyTorch nao esta disponivel no worker: "
            f"{runtime['torch_import_error'] or 'erro desconhecido'}."
        )

    if not runtime["cuda_available"]:
        detail = runtime["cuda_check_error"] or "torch.cuda.is_available() retornou false."
        raise RuntimeError(f"{operation} exige GPU CUDA, mas a GPU nao esta disponivel neste worker: {detail}")

    if runtime["device_count"] <= 0:
        raise RuntimeError(f"{operation} exige GPU CUDA, mas nenhuma GPU foi detectada pelo PyTorch.")

    primary_device = device.split(",")[0].strip()
    if primary_device.isdigit():
        device_index = int(primary_device)
        if device_index < 0 or device_index >= runtime["device_count"]:
            raise RuntimeError(
                f"{operation} exige GPU CUDA, mas o device '{device}' nao existe neste worker. "
                f"GPUs disponiveis: 0..{runtime['device_count'] - 1}."
            )

    return device

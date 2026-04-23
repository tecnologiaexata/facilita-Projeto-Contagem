#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
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


load_env_file(REPO_ROOT / ".env")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.annotation import build_color_mask, build_overlay  # noqa: E402
from app.services.yolo_segmentation import (  # noqa: E402
    ensure_ultralytics_available,
    predict_sample_class_mask,
    resolve_prediction_imgsz,
    resolve_training_params,
    resolve_yolo_model_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa inferencia local com um checkpoint YOLO .pt.")
    parser.add_argument("--image", required=True, help="Caminho local da imagem de entrada.")
    parser.add_argument(
        "--model",
        default="",
        help="Checkpoint .pt local. Se omitido, usa WORKER_DEFAULT_YOLO_MODEL do .env.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Pasta de saida. Se omitido, usa <imagem>_inference ao lado da imagem.",
    )
    parser.add_argument("--device", default="", help="Device do YOLO, por exemplo 0, cuda:0 ou cpu.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold.")
    parser.add_argument("--imgsz", default="", help="Image size, ex: 1280 ou 1536x2048.")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size opcional.")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Tile overlap opcional.")
    parser.add_argument("--native-resolution", action="store_true", help="Usa resolucao nativa do pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Imagem nao encontrada: {image_path}")

    ensure_ultralytics_available()
    from ultralytics import YOLO

    context: dict = {"training": {}}
    training_context = context["training"]
    if args.model:
        training_context["base_model"] = args.model
    if args.device:
        training_context["device"] = args.device
    if args.conf is not None:
        training_context["conf"] = args.conf
    if args.iou is not None:
        training_context["iou"] = args.iou
    if args.imgsz:
        training_context["imgsz"] = args.imgsz
    if args.tile_size is not None:
        training_context["tile_enabled"] = True
        training_context["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        training_context["tile_overlap"] = args.tile_overlap
    if args.native_resolution:
        training_context["native_resolution"] = True

    params = resolve_training_params(context)
    model_path = resolve_yolo_model_reference(args.model or params.get("model"))
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise SystemExit(
            "Modelo YOLO nao encontrado. Informe --model ou configure WORKER_DEFAULT_YOLO_MODEL no .env.\n"
            f"Valor resolvido: {model_path}"
        )

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)

    yolo = YOLO(str(resolved_model_path))
    prediction = predict_sample_class_mask(yolo, image_rgb, params=params, device=params.get("device"))
    predict_imgsz = resolve_prediction_imgsz(params, image_rgb.shape[:2])

    color_mask = build_color_mask(prediction)
    overlay = build_overlay(image_rgb, prediction)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else image_path.with_suffix("")
    if not args.output_dir:
        output_dir = output_dir.parent / f"{output_dir.name}_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "mask.png"
    color_mask_path = output_dir / "color-mask.png"
    overlay_path = output_dir / "overlay.png"

    Image.fromarray(prediction, mode="L").save(mask_path)
    Image.fromarray(color_mask).save(color_mask_path)
    Image.fromarray(overlay).save(overlay_path)

    print(f"model={resolved_model_path}")
    print(f"image={image_path}")
    print(f"imgsz={predict_imgsz}")
    print(f"device={params.get('device')}")
    print(f"mask={mask_path}")
    print(f"color_mask={color_mask_path}")
    print(f"overlay={overlay_path}")


if __name__ == "__main__":
    main()

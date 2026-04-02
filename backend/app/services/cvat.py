import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from app.config import ANNOTATED_CLASS_IDS, CLASS_MAP
from app.services.storage import ensure_directory


def contour_to_points(contour: np.ndarray) -> str:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, perimeter * 0.003)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = []
    for point in approx[:, 0, :]:
        x, y = point.tolist()
        points.append(f"{x:.1f},{y:.1f}")
    return ";".join(points)


def export_cvat_for_mask(
    sample_id: str,
    original_filename: str,
    class_mask: np.ndarray,
    width: int,
    height: int,
    destination: Path | None = None,
) -> bytes:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = f"facilita-coffee-{sample_id}"
    ET.SubElement(task, "size").text = "1"
    labels = ET.SubElement(task, "labels")
    for class_id in ANNOTATED_CLASS_IDS:
        class_meta = CLASS_MAP[class_id]
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = class_meta["slug"]
        ET.SubElement(label, "color").text = "#{:02x}{:02x}{:02x}".format(*class_meta["color"])
        ET.SubElement(label, "type").text = "polygon"
        ET.SubElement(label, "attributes")

    image = ET.SubElement(
        root,
        "image",
        {
            "id": "0",
            "name": original_filename,
            "width": str(width),
            "height": str(height),
        },
    )

    for class_id in ANNOTATED_CLASS_IDS:
        class_meta = CLASS_MAP[class_id]
        binary_mask = (class_mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 24:
                continue
            points = contour_to_points(contour)
            if points.count(";") < 2:
                continue
            ET.SubElement(
                image,
                "polygon",
                {
                    "label": class_meta["slug"],
                    "source": "manual",
                    "occluded": "0",
                    "points": points,
                    "z_order": "0",
                },
            )

    buffer = BytesIO()
    tree = ET.ElementTree(root)
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    payload = buffer.getvalue()

    if destination is not None:
        ensure_directory(destination.parent)
        destination.write_bytes(payload)

    return payload

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .geometry import (
    contour_area,
    contour_perimeter,
    orientation,
    pixel_to_mm,
    recommended_straightening_tolerance,
    remove_duplicate_points,
    simplify_points,
    smooth_points,
    straighten_points,
)
from .models import CNCContour, ProcessingParameters

logger = logging.getLogger(__name__)


def _normalize_foreground(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("A imagem para extração de contornos deve ter um canal.")
    mask = np.where(image > 0, 255, 0).astype(np.uint8)
    border = np.concatenate(
        (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    )
    if np.mean(border == 255) > 0.5 and np.mean(mask == 255) > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def _hierarchy_level(index: int, hierarchy: np.ndarray) -> int:
    level = 0
    parent = int(hierarchy[index][3])
    visited: set[int] = set()
    while parent >= 0 and parent not in visited:
        visited.add(parent)
        level += 1
        parent = int(hierarchy[parent][3])
    return level


def extract_contours(
    source_image: np.ndarray,
    pixels_per_mm: float,
    params: ProcessingParameters,
) -> list[CNCContour]:
    """Extrai contornos com RETR_TREE e converte o sistema para milímetros."""
    if pixels_per_mm <= 0:
        raise ValueError("Defina uma escala válida antes de extrair contornos.")
    mask = _normalize_foreground(source_image)
    if params.close_gaps and params.closing_tolerance_mm > 0:
        radius_px = max(
            1, int(round(params.closing_tolerance_mm * pixels_per_mm))
        )
        kernel_size = min(101, radius_px * 2 + 1)
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
    raw_contours, raw_hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if raw_hierarchy is None:
        return []
    hierarchy = raw_hierarchy[0]
    image_height = source_image.shape[0]
    min_area = params.min_area_mm2 if params.ignore_small_noise else 0.0
    min_perimeter = params.min_perimeter_mm if params.ignore_small_noise else 0.0

    candidates: list[tuple[int, CNCContour]] = []
    for source_index, raw in enumerate(raw_contours):
        raw_points = raw.reshape(-1, 2)
        points_mm = [
            pixel_to_mm(
                (float(point[0]), float(point[1])),
                pixels_per_mm,
                image_height,
            )
            for point in raw_points
        ]
        points_mm = remove_duplicate_points(points_mm)
        if len(points_mm) < 3:
            continue
        level = _hierarchy_level(source_index, hierarchy)
        is_hole = level % 2 == 1
        if is_hole and not params.keep_holes:
            continue
        area = contour_area(points_mm)
        perimeter = contour_perimeter(points_mm)
        if area < min_area or perimeter < min_perimeter:
            continue
        if params.smooth:
            points_mm = smooth_points(
                points_mm, params.smoothing_window, closed=True
            )
        if params.simplify:
            points_mm = simplify_points(
                points_mm, params.simplify_tolerance_mm, closed=True
            )
        if params.straighten_lines:
            straightening_tolerance = (
                recommended_straightening_tolerance(pixels_per_mm)
                if params.straighten_auto_tolerance
                else params.straighten_tolerance_mm
            )
            points_mm = straighten_points(
                points_mm,
                straightening_tolerance,
                closed=True,
                minimum_straight_length_mm=(
                    params.minimum_straight_length_mm
                    if params.preserve_rounded_sections
                    else 0.0
                ),
            )
        if len(points_mm) < 3:
            continue
        candidates.append(
            (
                source_index,
                CNCContour(
                    contour_id="",
                    points_mm=points_mm,
                    closed=True,
                    is_hole=is_hole,
                    hierarchy_level=level,
                    source_index=source_index,
                ),
            )
        )

    if params.keep_largest_only and candidates:
        outer_candidates = [
            item for item in candidates if not item[1].is_hole
        ] or candidates
        largest_index, _ = max(
            outer_candidates, key=lambda item: contour_area(item[1].points_mm)
        )

        def belongs_to_largest(index: int) -> bool:
            cursor = index
            while cursor >= 0:
                if cursor == largest_index:
                    return True
                cursor = int(hierarchy[cursor][3])
            return False

        candidates = [
            item for item in candidates if belongs_to_largest(item[0])
        ]

    id_by_source: dict[int, str] = {}
    for number, (source_index, contour) in enumerate(candidates, start=1):
        contour.contour_id = f"C{number:03d}"
        id_by_source[source_index] = contour.contour_id
    for source_index, contour in candidates:
        parent_index = int(hierarchy[source_index][3])
        while parent_index >= 0 and parent_index not in id_by_source:
            parent_index = int(hierarchy[parent_index][3])
        contour.parent_id = id_by_source.get(parent_index)

    result = [contour for _, contour in candidates]
    logger.info(
        "Extração concluída: %d contornos, %d furos.",
        len(result),
        sum(contour.is_hole for contour in result),
    )
    return result


def contour_summary(contour: CNCContour) -> dict[str, object]:
    return {
        "Selecionado": contour.selected,
        "ID": contour.contour_id,
        "Tipo": "furo" if contour.is_hole else "externo/ilha",
        "Área (mm²)": round(contour_area(contour.points_mm, contour.closed), 3),
        "Perímetro (mm)": round(
            contour_perimeter(contour.points_mm, contour.closed), 3
        ),
        "Pontos": len(contour.points_mm),
        "Fechado": contour.closed,
        "Nível": contour.hierarchy_level,
        "Sentido": orientation(contour.points_mm),
    }


def close_contour_if_near(contour: CNCContour, tolerance_mm: float) -> CNCContour:
    if contour.closed or len(contour.points_mm) < 3:
        return contour
    if math.dist(contour.points_mm[0], contour.points_mm[-1]) <= tolerance_mm:
        contour.closed = True
    return contour

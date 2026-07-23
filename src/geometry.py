from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon

from .models import CNCContour, Point


def pixels_per_mm_from_width(width_px: float, width_mm: float) -> float:
    if width_px <= 0 or width_mm <= 0:
        raise ValueError("Largura em pixels e milímetros devem ser positivas.")
    return float(width_px) / float(width_mm)


def pixels_per_mm_from_points(
    point_1_px: Point, point_2_px: Point, distance_mm: float
) -> float:
    pixel_distance = math.dist(point_1_px, point_2_px)
    if pixel_distance <= 0 or distance_mm <= 0:
        raise ValueError("Os pontos devem ser distintos e a distância deve ser positiva.")
    return pixel_distance / float(distance_mm)


def pixel_to_mm(point_px: Point, pixels_per_mm: float, image_height_px: int = 0) -> Point:
    if pixels_per_mm <= 0:
        raise ValueError("A escala deve ser maior que zero.")
    x, y = point_px
    cartesian_y = image_height_px - y if image_height_px else y
    return float(x) / pixels_per_mm, float(cartesian_y) / pixels_per_mm


def apply_scale(points_px: Iterable[Point], pixels_per_mm: float) -> list[Point]:
    if pixels_per_mm <= 0:
        raise ValueError("A escala deve ser maior que zero.")
    return [(float(x) / pixels_per_mm, float(y) / pixels_per_mm) for x, y in points_px]


def bounding_box(contours: Iterable[CNCContour]) -> tuple[float, float, float, float]:
    points = [
        point
        for contour in contours
        if contour.selected
        for point in contour.points_mm
    ]
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def dimensions(contours: Iterable[CNCContour]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bounding_box(contours)
    return max_x - min_x, max_y - min_y


def remove_duplicate_points(
    points: Iterable[Point], tolerance: float = 1e-9
) -> list[Point]:
    result: list[Point] = []
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if not result or math.dist(result[-1], normalized) > tolerance:
            result.append(normalized)
    if len(result) > 1 and math.dist(result[0], result[-1]) <= tolerance:
        result.pop()
    return result


def is_contour_closed(
    points: list[Point], tolerance: float = 1e-6, closed_flag: bool | None = None
) -> bool:
    if closed_flag is not None:
        return closed_flag and len(points) >= 3
    return len(points) >= 3 and math.dist(points[0], points[-1]) <= tolerance


def signed_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )


def orientation(points: list[Point]) -> str:
    return "anti-horário" if signed_area(points) > 0 else "horário"


def contour_area(points: list[Point], closed: bool = True) -> float:
    return abs(signed_area(points)) if closed and len(points) >= 3 else 0.0


def contour_perimeter(points: list[Point], closed: bool = True) -> float:
    if len(points) < 2:
        return 0.0
    pairs = list(zip(points, points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    return sum(math.dist(first, second) for first, second in pairs)


def simplify_points(points: list[Point], tolerance_mm: float, closed: bool = True) -> list[Point]:
    if tolerance_mm <= 0 or len(points) < 3:
        return list(points)
    array = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(array, float(tolerance_mm), closed)
    return remove_duplicate_points(
        [(float(item[0][0]), float(item[0][1])) for item in simplified]
    )


def straighten_points(
    points: list[Point],
    tolerance_mm: float,
    closed: bool = True,
    max_area_change_ratio: float = 0.05,
) -> list[Point]:
    """Substitui ruído raster quase linear por segmentos retos controlados.

    O desvio máximo é limitado por ``tolerance_mm``. Para contornos fechados, uma
    aproximação inválida ou que altere mais de 5% da área é rejeitada.
    """
    if tolerance_mm <= 0 or len(points) < 3:
        return list(points)
    straightened = simplify_points(points, tolerance_mm, closed)
    minimum_points = 3 if closed else 2
    if len(straightened) < minimum_points:
        return list(points)
    if closed:
        original = Polygon(points)
        candidate = Polygon(straightened)
        if (
            original.is_empty
            or not original.is_valid
            or candidate.is_empty
            or not candidate.is_valid
        ):
            return list(points)
        if original.area > 0:
            area_change = abs(candidate.area - original.area) / original.area
            if area_change > max_area_change_ratio:
                return list(points)
    return straightened


def recommended_straightening_tolerance(pixels_per_mm: float) -> float:
    """Retorna tolerância equivalente a dois pixels, entre 0,2 e 3 mm."""
    if pixels_per_mm <= 0:
        raise ValueError("A escala deve ser maior que zero.")
    return max(0.2, min(3.0, 2.0 / pixels_per_mm))


def smooth_points(points: list[Point], window: int = 3, closed: bool = True) -> list[Point]:
    if len(points) < 4 or window < 2:
        return list(points)
    window = min(int(window), len(points) - 1)
    result: list[Point] = []
    for index in range(len(points)):
        if closed:
            sample = [
                points[(index + offset) % len(points)]
                for offset in range(-(window // 2), window // 2 + 1)
            ]
        else:
            start = max(0, index - window // 2)
            end = min(len(points), index + window // 2 + 1)
            sample = points[start:end]
        result.append(
            (
                sum(point[0] for point in sample) / len(sample),
                sum(point[1] for point in sample) / len(sample),
            )
        )
    return result


def has_self_intersection(points: list[Point], closed: bool = True) -> bool:
    if len(points) < 4:
        return False
    geometry = Polygon(points) if closed else LineString(points)
    return not geometry.is_valid or (not closed and not geometry.is_simple)


def transform_contours(
    contours: list[CNCContour],
    *,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    rotate_degrees: float = 0.0,
    mirror_horizontal: bool = False,
    mirror_vertical: bool = False,
    origin: Point = (0.0, 0.0),
) -> list[CNCContour]:
    result = deepcopy(contours)
    angle = math.radians(rotate_degrees)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    ox, oy = origin
    for contour in result:
        def transform_points(points: list[Point]) -> list[Point]:
            transformed: list[Point] = []
            for x, y in points:
                local_x, local_y = x - ox, y - oy
                if mirror_horizontal:
                    local_x *= -1
                if mirror_vertical:
                    local_y *= -1
                rotated_x = local_x * cos_angle - local_y * sin_angle
                rotated_y = local_x * sin_angle + local_y * cos_angle
                transformed.append(
                    (
                        rotated_x + ox + translate_x,
                        rotated_y + oy + translate_y,
                    )
                )
            return transformed

        contour.points_mm = transform_points(contour.points_mm)
        if contour.uncompensated_points_mm is not None:
            contour.uncompensated_points_mm = transform_points(
                contour.uncompensated_points_mm
            )
    return result


def align_lower_left(contours: list[CNCContour], target: Point = (0.0, 0.0)) -> list[CNCContour]:
    min_x, min_y, _, _ = bounding_box(contours)
    return transform_contours(
        contours,
        translate_x=target[0] - min_x,
        translate_y=target[1] - min_y,
    )


def center_at_origin(contours: list[CNCContour]) -> list[CNCContour]:
    min_x, min_y, max_x, max_y = bounding_box(contours)
    return transform_contours(
        contours,
        translate_x=-(min_x + max_x) / 2,
        translate_y=-(min_y + max_y) / 2,
    )


def reverse_contour(contour: CNCContour) -> CNCContour:
    result = deepcopy(contour)
    result.points_mm = list(reversed(result.points_mm))
    if result.uncompensated_points_mm is not None:
        result.uncompensated_points_mm = list(
            reversed(result.uncompensated_points_mm)
        )
    return result


def offset_contour(contour: CNCContour, distance_mm: float) -> CNCContour:
    if not contour.closed or len(contour.points_mm) < 3:
        raise ValueError("Somente contornos fechados podem receber compensação.")
    polygon = Polygon(contour.points_mm)
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("O contorno original é inválido.")
    buffered = polygon.buffer(float(distance_mm), join_style=2)
    if buffered.is_empty:
        raise ValueError("A compensação destruiu a geometria.")
    if isinstance(buffered, MultiPolygon):
        buffered = max(buffered.geoms, key=lambda geometry: geometry.area)
    if not isinstance(buffered, Polygon):
        raise ValueError("A compensação não produziu um polígono utilizável.")
    result = deepcopy(contour)
    if result.uncompensated_points_mm is None:
        result.uncompensated_points_mm = list(contour.points_mm)
    result.points_mm = remove_duplicate_points(
        [(float(x), float(y)) for x, y in list(buffered.exterior.coords)]
    )
    result.compensation_mm += float(distance_mm)
    return result

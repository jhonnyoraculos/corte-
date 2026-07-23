from __future__ import annotations

import math

import pytest

from src.geometry import (
    apply_scale,
    bounding_box,
    dimensions,
    has_self_intersection,
    is_contour_closed,
    offset_contour,
    pixel_to_mm,
    pixels_per_mm_from_points,
    pixels_per_mm_from_width,
    recommended_straightening_tolerance,
    remove_duplicate_points,
    simplify_points,
    straighten_points,
)
from tests.helpers import rectangle_contour


def test_pixel_to_millimeter_conversion() -> None:
    assert pixel_to_mm((20, 10), 2.0) == (10.0, 5.0)
    assert pixel_to_mm((20, 10), 2.0, image_height_px=100) == (10.0, 45.0)


def test_scale_calculation_and_application() -> None:
    assert pixels_per_mm_from_width(1000, 500) == 2.0
    assert pixels_per_mm_from_points((0, 0), (300, 400), 250) == 2.0
    assert apply_scale([(0, 0), (100, 50)], 2.0) == [
        (0.0, 0.0),
        (50.0, 25.0),
    ]


def test_bounding_box_and_dimensions() -> None:
    contour = rectangle_contour(100, 50)
    assert bounding_box([contour]) == (0.0, 0.0, 100.0, 50.0)
    assert dimensions([contour]) == (100.0, 50.0)


def test_closed_contour_detection() -> None:
    closed = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 0.0)]
    open_line = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
    assert is_contour_closed(closed)
    assert not is_contour_closed(open_line)
    assert is_contour_closed(open_line, closed_flag=True)


def test_duplicate_point_removal() -> None:
    points = [(0, 0), (0, 0), (1, 0), (1, 1), (0, 0)]
    assert remove_duplicate_points(points) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]


def test_polygon_simplification_preserves_shape() -> None:
    points = [
        (0, 0),
        (1, 0.001),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2.001),
        (0, 2),
        (0, 1),
    ]
    simplified = simplify_points(points, 0.01)
    assert len(simplified) == 4
    assert bounding_box(
        [
            rectangle_contour()
            if not simplified
            else type(rectangle_contour())(
                "C",
                simplified,
                True,
                False,
                0,
            )
        ]
    )[2] == pytest.approx(2.0)


def test_straightening_replaces_raster_wobble_with_one_straight_segment() -> None:
    noisy_polygon = [
        (0.0, 0.0),
        (5.0, 5.8),
        (10.0, 9.2),
        (15.0, 15.7),
        (20.0, 19.1),
        (25.0, 25.5),
        (30.0, 30.0),
        (30.0, 60.0),
        (0.0, 60.0),
    ]
    straightened = straighten_points(noisy_polygon, 2.0)
    assert straightened == [
        (0.0, 0.0),
        (30.0, 30.0),
        (30.0, 60.0),
        (0.0, 60.0),
    ]


def test_straightening_preserves_a_rounded_contour() -> None:
    circle = [
        (
            50.0 + 50.0 * math.cos(math.radians(angle)),
            50.0 + 50.0 * math.sin(math.radians(angle)),
        )
        for angle in range(0, 360, 2)
    ]
    straightened = straighten_points(circle, 2.0)
    assert 8 <= len(straightened) < len(circle)
    xs = [point[0] for point in straightened]
    ys = [point[1] for point in straightened]
    assert max(xs) - min(xs) == pytest.approx(100.0, abs=2.0)
    assert max(ys) - min(ys) == pytest.approx(100.0, abs=2.0)


def test_straightening_tolerance_adapts_to_image_resolution() -> None:
    assert recommended_straightening_tolerance(1.0) == 2.0
    assert recommended_straightening_tolerance(10.0) == 0.2
    assert recommended_straightening_tolerance(0.1) == 3.0


def test_self_intersection_detection() -> None:
    bow_tie = [(0, 0), (2, 2), (0, 2), (2, 0)]
    assert has_self_intersection(bow_tie)
    assert not has_self_intersection(rectangle_contour().points_mm)


def test_internal_tool_compensation() -> None:
    original = rectangle_contour(100, 50)
    compensated = offset_contour(original, -2)
    assert dimensions([compensated]) == pytest.approx((96.0, 46.0))
    assert compensated.compensation_mm == -2


def test_external_tool_compensation() -> None:
    original = rectangle_contour(100, 50)
    compensated = offset_contour(original, 2)
    assert dimensions([compensated]) == pytest.approx((104.0, 54.0))
    assert compensated.compensation_mm == 2

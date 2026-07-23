from __future__ import annotations

from src.models import CNCContour, CNCProject, Calibration


def rectangle_contour(
    width: float = 100.0,
    height: float = 50.0,
    *,
    contour_id: str = "C001",
    is_hole: bool = False,
) -> CNCContour:
    return CNCContour(
        contour_id=contour_id,
        points_mm=[(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
        closed=True,
        is_hole=is_hole,
        hierarchy_level=1 if is_hole else 0,
    )


def valid_project() -> CNCProject:
    return CNCProject(
        name="teste",
        calibration=Calibration(method="known_width", pixels_per_mm=2.0),
        contours=[rectangle_contour()],
    )

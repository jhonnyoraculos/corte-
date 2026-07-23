from __future__ import annotations

from src.models import CNCContour, Calibration, CNCProject, MachineProfile, ValidationSeverity
from src.validators import has_critical_errors, validate_project
from tests.helpers import rectangle_contour, valid_project


def test_valid_project_has_no_critical_geometry_errors() -> None:
    project = valid_project()
    project.machine_profile = MachineProfile(profile_name="Teste")
    issues = validate_project(project)
    assert not has_critical_errors(issues)


def test_self_intersection_and_negative_coordinates_are_errors() -> None:
    contour = CNCContour(
        "C001",
        [(-1, 0), (2, 2), (0, 2), (2, 0)],
        True,
        False,
        0,
    )
    project = CNCProject(
        calibration=Calibration(method="known_width", pixels_per_mm=1),
        contours=[contour],
    )
    issues = validate_project(project)
    error_codes = {
        issue.code
        for issue in issues
        if issue.severity == ValidationSeverity.ERROR
    }
    assert "invalid_polygon" in error_codes
    assert "negative_coordinates" in error_codes


def test_hole_outside_external_contour_is_rejected() -> None:
    outer = rectangle_contour(100, 50)
    hole = CNCContour(
        "C002",
        [(150, 10), (160, 10), (160, 20), (150, 20)],
        True,
        True,
        1,
        parent_id="C001",
    )
    project = CNCProject(
        calibration=Calibration(method="known_width", pixels_per_mm=1),
        contours=[outer, hole],
    )
    issues = validate_project(project)
    assert any(issue.code == "hole_outside" for issue in issues)


def test_machine_limits_are_enforced() -> None:
    project = valid_project()
    project.machine_profile = MachineProfile(
        profile_name="Máquina pequena",
        max_width_mm=80,
        max_height_mm=40,
        max_depth_mm=2,
    )
    issues = validate_project(project)
    codes = {issue.code for issue in issues}
    assert {"machine_width", "machine_height", "machine_depth"} <= codes

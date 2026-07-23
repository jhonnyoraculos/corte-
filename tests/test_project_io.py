from __future__ import annotations

import math

import pytest

from src.geometry import offset_contour
from src.models import CNCProject, MachineProfile
from tests.helpers import valid_project


def test_project_json_round_trip_preserves_vectors_and_compensation() -> None:
    project = valid_project()
    project.contours = [offset_contour(project.contours[0], 2.0)]
    project.machine_profile = MachineProfile(profile_name="Perfil salvo")
    loaded = CNCProject.from_json(project.to_json())
    assert loaded.name == project.name
    assert loaded.calibration.pixels_per_mm == 2.0
    assert loaded.contours[0].points_mm == project.contours[0].points_mm
    assert loaded.contours[0].uncompensated_points_mm == [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 50.0),
        (0.0, 50.0),
    ]
    assert loaded.machine_profile is not None
    assert loaded.machine_profile.profile_name == "Perfil salvo"


def test_malformed_profile_types_are_rejected() -> None:
    with pytest.raises(ValueError):
        MachineProfile.from_dict(
            {
                "profile_name": "Inválido",
                "coordinate_precision": "três",
            }
        )


def test_non_finite_project_value_is_rejected() -> None:
    data = valid_project().to_dict()
    data["material_thickness_mm"] = math.nan
    with pytest.raises(ValueError):
        CNCProject.from_dict(data)

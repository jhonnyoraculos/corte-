from __future__ import annotations

import io

import ezdxf
from ezdxf import units
import pytest

from src.exporters.dxf_exporter import DXFExportOptions, generate_dxf_bytes
from tests.helpers import valid_project


def read_dxf(data: bytes):
    return ezdxf.read(io.StringIO(data.decode("utf-8")))


@pytest.mark.parametrize(
    ("version", "entity_type"),
    [("R12", "POLYLINE"), ("R2000", "LWPOLYLINE")],
)
def test_dxf_versions_are_valid_and_closed(version: str, entity_type: str) -> None:
    project = valid_project()
    data = generate_dxf_bytes(
        project,
        DXFExportOptions(
            version=version,
            include_reference=False,
            include_start_points=False,
        ),
    )
    doc = read_dxf(data)
    entities = list(doc.modelspace().query(entity_type))
    assert len(entities) == 1
    assert entities[0].is_closed
    if version == "R2000":
        assert doc.units == units.MM
    else:
        # O DXF R12 não possui $INSUNITS; a unidade fica explícita nos metadados.
        metadata = [entity.dxf.text for entity in doc.modelspace().query("TEXT")]
        assert any("Unidade=mm" in text for text in metadata)
    assert "CUT_OUTER" in doc.layers
    assert "CUT_INNER" in doc.layers


@pytest.mark.parametrize("version", ["R12", "R2000"])
def test_dxf_preserves_coordinates_and_dimensions(version: str) -> None:
    data = generate_dxf_bytes(
        valid_project(),
        DXFExportOptions(
            version=version,
            include_reference=False,
            include_start_points=False,
        ),
    )
    doc = read_dxf(data)
    if version == "R12":
        entity = doc.modelspace().query("POLYLINE").first
        points = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
    else:
        entity = doc.modelspace().query("LWPOLYLINE").first
        points = [(point[0], point[1]) for point in entity.get_points("xy")]
    assert min(x for x, _ in points) == pytest.approx(0)
    assert max(x for x, _ in points) == pytest.approx(100)
    assert min(y for _, y in points) == pytest.approx(0)
    assert max(y for _, y in points) == pytest.approx(50)

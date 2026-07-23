from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import logging

import ezdxf
from ezdxf import units

from ..geometry import bounding_box, dimensions
from ..models import CNCProject
from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DXFExportOptions:
    version: str = "R2000"
    include_reference: bool = True
    include_start_points: bool = True


class DXFExporter(BaseExporter[bytes]):
    def __init__(self, options: DXFExportOptions | None = None) -> None:
        self.options = options or DXFExportOptions()

    def generate(self, project: CNCProject) -> bytes:
        contours = project.selected_contours
        if not contours:
            raise ValueError("Não há contornos selecionados para exportar.")
        if self.options.version not in {"R12", "R2000"}:
            raise ValueError("Versão DXF não suportada.")

        dxf_version = "R12" if self.options.version == "R12" else "R2000"
        doc = ezdxf.new(dxf_version, setup=True)
        doc.units = units.MM
        doc.header["$INSUNITS"] = units.MM
        doc.header["$MEASUREMENT"] = 1
        for layer, color in (
            ("CUT_OUTER", 5),
            ("CUT_INNER", 1),
            ("REFERENCE", 8),
            ("START_POINTS", 3),
        ):
            if layer not in doc.layers:
                doc.layers.add(layer, color=color)
        modelspace = doc.modelspace()
        for contour in contours:
            layer = "CUT_INNER" if contour.is_hole else "CUT_OUTER"
            if self.options.version == "R12":
                entity = modelspace.add_polyline2d(
                    contour.points_mm, dxfattribs={"layer": layer}
                )
                if contour.closed:
                    entity.close(True)
            else:
                modelspace.add_lwpolyline(
                    contour.points_mm,
                    close=contour.closed,
                    dxfattribs={"layer": layer},
                )
            if self.options.include_start_points and contour.points_mm:
                x, y = contour.points_mm[0]
                modelspace.add_point((x, y), dxfattribs={"layer": "START_POINTS"})

        min_x, min_y, max_x, max_y = bounding_box(contours)
        if self.options.include_reference:
            modelspace.add_point((0.0, 0.0), dxfattribs={"layer": "REFERENCE"})
            modelspace.add_lwpolyline(
                [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                ],
                close=True,
                dxfattribs={"layer": "REFERENCE"},
            ) if self.options.version != "R12" else _add_r12_reference(
                modelspace, min_x, min_y, max_x, max_y
            )

        width, height = dimensions(contours)
        metadata = (
            f"Projeto={project.name}; Unidade=mm; "
            f"Largura={width:.3f}; Altura={height:.3f}; "
            f"Gerado={datetime.now(timezone.utc).isoformat()}"
        )
        modelspace.add_text(
            metadata,
            dxfattribs={"layer": "REFERENCE", "height": max(width, height, 1.0) / 100},
        ).set_placement((min_x, min_y))

        binary_stream = io.BytesIO()
        text_stream = io.TextIOWrapper(
            binary_stream,
            encoding=doc.output_encoding,
            errors="dxfreplace",
            newline="",
        )
        doc.write(text_stream)
        text_stream.flush()
        data = binary_stream.getvalue()
        text_stream.detach()
        logger.info(
            "DXF %s gerado em memória com %d contornos.", self.options.version, len(contours)
        )
        return data


def _add_r12_reference(modelspace, min_x: float, min_y: float, max_x: float, max_y: float):
    entity = modelspace.add_polyline2d(
        [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ],
        dxfattribs={"layer": "REFERENCE"},
    )
    entity.close(True)
    return entity


def generate_dxf_bytes(
    project: CNCProject, options: DXFExportOptions | None = None
) -> bytes:
    return DXFExporter(options).generate(project)

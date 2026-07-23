from __future__ import annotations

from dataclasses import dataclass
import io

import ezdxf
import plotly.graph_objects as go

from .geometry import bounding_box, dimensions
from .models import CNCContour


@dataclass(slots=True)
class DXFPreviewResult:
    figure: go.Figure
    version: str
    entity_count: int
    polyline_count: int
    point_count: int
    layers: list[str]
    bounds: tuple[float, float, float, float]


def build_vector_preview(
    contours: list[CNCContour],
    *,
    show_points: bool = False,
    show_compensation: bool = True,
    show_ids: bool = True,
) -> go.Figure:
    selected = [contour for contour in contours if contour.selected]
    figure = go.Figure()
    for contour in selected:
        compensated = bool(
            contour.compensation_mm and contour.uncompensated_points_mm
        )
        points = list(
            contour.points_mm
            if show_compensation or not compensated
            else contour.uncompensated_points_mm or contour.points_mm
        )
        if contour.closed and points:
            points.append(points[0])
        if not points:
            continue
        xs, ys = zip(*points)
        color = "#ef4444" if contour.is_hole else "#2563eb"
        name = f"{contour.contour_id} — {'furo' if contour.is_hole else 'externo'}"
        dash = "dash" if show_compensation and compensated else "solid"
        if show_compensation and compensated:
            base_points = list(contour.uncompensated_points_mm or [])
            if contour.closed and base_points:
                base_points.append(base_points[0])
            base_x, base_y = zip(*base_points)
            figure.add_trace(
                go.Scatter(
                    x=base_x,
                    y=base_y,
                    mode="lines",
                    line={"color": "#94a3b8", "width": 1, "dash": "dot"},
                    name=f"Original {contour.contour_id}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers" if show_points else "lines",
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": 5},
                name=name,
                hovertemplate=f"{contour.contour_id}<br>X=%{{x:.3f}} mm<br>Y=%{{y:.3f}} mm<extra></extra>",
            )
        )
        start_x, start_y = contour.points_mm[0]
        figure.add_trace(
            go.Scatter(
                x=[start_x],
                y=[start_y],
                mode="markers+text" if show_ids else "markers",
                text=[contour.contour_id] if show_ids else None,
                textposition="top right",
                marker={"symbol": "diamond", "size": 9, "color": "#16a34a"},
                name=f"Início {contour.contour_id}",
                showlegend=False,
                hovertemplate="Ponto inicial<extra></extra>",
            )
        )
        if len(contour.points_mm) > 1:
            next_x, next_y = contour.points_mm[1]
            figure.add_annotation(
                x=next_x,
                y=next_y,
                ax=start_x,
                ay=start_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color,
                text="",
            )

    min_x, min_y, max_x, max_y = bounding_box(selected)
    width, height = dimensions(selected)
    if selected:
        figure.add_shape(
            type="rect",
            x0=min_x,
            y0=min_y,
            x1=max_x,
            y1=max_y,
            line={"color": "#64748b", "dash": "dot", "width": 1},
        )
        figure.add_annotation(
            x=(min_x + max_x) / 2,
            y=max_y,
            text=f"{width:.3f} × {height:.3f} mm",
            showarrow=False,
            yshift=14,
        )
    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            text=["Origem"],
            textposition="bottom right",
            marker={"symbol": "cross", "size": 12, "color": "#111827"},
            name="Origem X=0, Y=0",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=650,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        legend={"orientation": "h"},
        hovermode="closest",
    )
    figure.update_xaxes(title="X (mm)", scaleanchor="y", scaleratio=1)
    figure.update_yaxes(title="Y (mm)")
    return figure


def _read_dxf_bytes(data: bytes):
    if not data:
        raise ValueError("O DXF gerado está vazio.")
    last_error: Exception | None = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return ezdxf.read(io.StringIO(data.decode(encoding)))
        except (UnicodeDecodeError, ezdxf.DXFError, ValueError) as exc:
            last_error = exc
    raise ValueError("Não foi possível reler o DXF gerado.") from last_error


def build_dxf_preview(data: bytes) -> DXFPreviewResult:
    """Relê o arquivo DXF final e desenha as entidades efetivamente exportadas."""
    document = _read_dxf_bytes(data)
    modelspace = document.modelspace()
    figure = go.Figure()
    colors = {
        "CUT_OUTER": "#2563eb",
        "CUT_INNER": "#ef4444",
        "REFERENCE": "#64748b",
        "START_POINTS": "#16a34a",
    }
    seen_legends: set[tuple[str, str]] = set()
    layers: set[str] = set()
    all_points: list[tuple[float, float]] = []
    entity_count = 0
    polyline_count = 0
    point_count = 0

    for entity in modelspace:
        entity_count += 1
        entity_type = entity.dxftype()
        layer = str(entity.dxf.layer)
        layers.add(layer)
        color = colors.get(layer, "#7c3aed")

        points: list[tuple[float, float]] = []
        closed = False
        if entity_type == "LWPOLYLINE":
            points = [
                (float(point[0]), float(point[1]))
                for point in entity.get_points("xy")
            ]
            closed = bool(entity.closed)
        elif entity_type == "POLYLINE":
            points = [
                (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
                for vertex in entity.vertices
            ]
            closed = bool(entity.is_closed)
        elif entity_type == "LINE":
            points = [
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            ]

        if points:
            polyline_count += 1
            all_points.extend(points)
            displayed = points + [points[0]] if closed and len(points) > 1 else points
            xs, ys = zip(*displayed)
            legend_key = (layer, "line")
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={
                        "color": color,
                        "width": 2,
                        "dash": "dot" if layer == "REFERENCE" else "solid",
                    },
                    name=layer,
                    legendgroup=layer,
                    showlegend=legend_key not in seen_legends,
                    hovertemplate=(
                        f"Camada: {layer}<br>Entidade: {entity_type}"
                        "<br>X=%{x:.3f} mm<br>Y=%{y:.3f} mm<extra></extra>"
                    ),
                )
            )
            seen_legends.add(legend_key)
        elif entity_type == "POINT":
            point = (float(entity.dxf.location.x), float(entity.dxf.location.y))
            all_points.append(point)
            point_count += 1
            legend_key = (layer, "point")
            figure.add_trace(
                go.Scatter(
                    x=[point[0]],
                    y=[point[1]],
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 9,
                        "symbol": "diamond" if layer == "START_POINTS" else "cross",
                    },
                    name=layer,
                    legendgroup=layer,
                    showlegend=legend_key not in seen_legends,
                    hovertemplate=(
                        f"Camada: {layer}<br>Entidade: POINT"
                        "<br>X=%{x:.3f} mm<br>Y=%{y:.3f} mm<extra></extra>"
                    ),
                )
            )
            seen_legends.add(legend_key)

    if not all_points:
        raise ValueError("O DXF não contém entidades geométricas visualizáveis.")
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    figure.update_layout(
        title=f"DXF relido — {width:.3f} × {height:.3f} mm",
        template="plotly_white",
        height=600,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h"},
        hovermode="closest",
    )
    figure.update_xaxes(title="X (mm)", scaleanchor="y", scaleratio=1)
    figure.update_yaxes(title="Y (mm)")
    return DXFPreviewResult(
        figure=figure,
        version=document.acad_release,
        entity_count=entity_count,
        polyline_count=polyline_count,
        point_count=point_count,
        layers=sorted(layers),
        bounds=bounds,
    )

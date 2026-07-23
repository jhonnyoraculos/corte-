from __future__ import annotations

import plotly.graph_objects as go

from .geometry import bounding_box, dimensions
from .models import CNCContour


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

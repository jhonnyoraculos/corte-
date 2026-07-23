from __future__ import annotations

import math
from collections import Counter

from itertools import combinations

from shapely.geometry import Polygon

from .geometry import (
    bounding_box,
    dimensions,
    has_self_intersection,
)
from .models import (
    CNCProject,
    MachineProfile,
    ValidationIssue,
    ValidationSeverity,
)


def _issue(
    code: str,
    message: str,
    severity: ValidationSeverity,
    contour_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(code, message, severity, contour_id)


def validate_project(
    project: CNCProject, segment_tolerance_mm: float = 0.01
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not project.calibration.is_defined:
        issues.append(
            _issue(
                "scale_missing",
                "A escala em pixels por milímetro ainda não foi definida.",
                ValidationSeverity.ERROR,
            )
        )
    selected = project.selected_contours
    if not selected:
        issues.append(
            _issue(
                "no_contours",
                "Selecione pelo menos um contorno para exportação.",
                ValidationSeverity.ERROR,
            )
        )
        return issues

    width, height = dimensions(selected)
    if width <= 0 or height <= 0:
        issues.append(
            _issue(
                "invalid_dimensions",
                "As dimensões finais devem ser maiores que zero.",
                ValidationSeverity.ERROR,
            )
        )
    else:
        issues.append(
            _issue(
                "dimensions",
                f"Dimensões vetoriais: {width:.3f} × {height:.3f} mm.",
                ValidationSeverity.INFO,
            )
        )

    outer_polygons: list[Polygon] = []
    valid_polygons: list[tuple[str, Polygon]] = []
    for contour in selected:
        points = contour.points_mm
        if len(points) < 2:
            issues.append(
                _issue(
                    "empty_geometry",
                    "O contorno não possui pontos suficientes.",
                    ValidationSeverity.ERROR,
                    contour.contour_id,
                )
            )
            continue
        rounded = [(round(x, 9), round(y, 9)) for x, y in points]
        duplicates = [point for point, count in Counter(rounded).items() if count > 1]
        if duplicates:
            issues.append(
                _issue(
                    "duplicate_points",
                    f"Foram encontrados {len(duplicates)} pontos duplicados.",
                    ValidationSeverity.WARNING,
                    contour.contour_id,
                )
            )
        pairs = list(zip(points, points[1:]))
        if contour.closed:
            pairs.append((points[-1], points[0]))
        short_count = sum(
            math.dist(first, second) < segment_tolerance_mm
            for first, second in pairs
        )
        if short_count:
            issues.append(
                _issue(
                    "short_segments",
                    f"{short_count} segmento(s) abaixo de {segment_tolerance_mm:g} mm.",
                    ValidationSeverity.WARNING,
                    contour.contour_id,
                )
            )
        if not contour.closed:
            issues.append(
                _issue(
                    "open_contour",
                    "O contorno está aberto.",
                    ValidationSeverity.ERROR,
                    contour.contour_id,
                )
            )
        elif len(points) >= 3:
            polygon = Polygon(points)
            if polygon.is_empty:
                issues.append(
                    _issue(
                        "empty_polygon",
                        "A geometria resultou em um polígono vazio.",
                        ValidationSeverity.ERROR,
                        contour.contour_id,
                    )
                )
            if not polygon.is_valid or has_self_intersection(points):
                issues.append(
                    _issue(
                        "invalid_polygon",
                        "O contorno possui auto-interseção ou polígono inválido.",
                        ValidationSeverity.ERROR,
                        contour.contour_id,
                    )
                )
            if contour.compensation_mm and (polygon.is_empty or polygon.area <= 0):
                issues.append(
                    _issue(
                        "compensation_destroyed",
                        "A compensação de ferramenta destruiu a geometria.",
                        ValidationSeverity.ERROR,
                        contour.contour_id,
                    )
                )
            if not contour.is_hole and polygon.is_valid:
                outer_polygons.append(polygon)
            if polygon.is_valid and not polygon.is_empty:
                valid_polygons.append((contour.contour_id, polygon))
        if len(points) > 10_000:
            issues.append(
                _issue(
                    "too_many_points",
                    "O contorno possui mais de 10.000 pontos; considere simplificá-lo.",
                    ValidationSeverity.WARNING,
                    contour.contour_id,
                )
            )

    for contour in [item for item in selected if item.is_hole and len(item.points_mm) >= 3]:
        hole = Polygon(contour.points_mm)
        if hole.is_valid and not any(
            outer.contains(hole.representative_point()) for outer in outer_polygons
        ):
            issues.append(
                _issue(
                    "hole_outside",
                    "O furo não está contido em nenhum contorno externo selecionado.",
                    ValidationSeverity.ERROR,
                    contour.contour_id,
                )
            )

    for (first_id, first), (second_id, second) in combinations(valid_polygons, 2):
        boundary_intersection = first.boundary.intersection(second.boundary)
        if not boundary_intersection.is_empty:
            issues.append(
                _issue(
                    "crossing_contours",
                    f"Os limites de {first_id} e {second_id} se cruzam ou se tocam.",
                    ValidationSeverity.ERROR,
                )
            )

    min_x, min_y, max_x, max_y = bounding_box(selected)
    profile = project.machine_profile
    if min_x < 0 or min_y < 0:
        severity = (
            ValidationSeverity.WARNING
            if profile and profile.allow_negative_coordinates
            else ValidationSeverity.ERROR
        )
        issues.append(
            _issue(
                "negative_coordinates",
                f"Há coordenadas negativas (mínimo X={min_x:.3f}, Y={min_y:.3f}).",
                severity,
            )
        )
    if profile:
        if profile.max_width_mm is not None and width > profile.max_width_mm:
            issues.append(
                _issue(
                    "machine_width",
                    "A largura excede a área útil configurada da máquina.",
                    ValidationSeverity.ERROR,
                )
            )
        if profile.max_height_mm is not None and height > profile.max_height_mm:
            issues.append(
                _issue(
                    "machine_height",
                    "A altura excede a área útil configurada da máquina.",
                    ValidationSeverity.ERROR,
                )
            )
        if (
            profile.max_width_mm is not None
            and (min_x < 0 or max_x > profile.max_width_mm)
        ) or (
            profile.max_height_mm is not None
            and (min_y < 0 or max_y > profile.max_height_mm)
        ):
            issues.append(
                _issue(
                    "machine_work_area",
                    "Há geometria fora da área útil configurada da máquina.",
                    ValidationSeverity.ERROR,
                )
            )
        if (
            profile.max_depth_mm is not None
            and project.tool.cutting_depth_mm > profile.max_depth_mm
        ):
            issues.append(
                _issue(
                    "machine_depth",
                    "A profundidade de corte excede o limite configurado.",
                    ValidationSeverity.ERROR,
                )
            )
        if not profile.supports_open_contours and any(
            not contour.closed for contour in selected
        ):
            issues.append(
                _issue(
                    "profile_open_contours",
                    "O perfil da máquina não permite contornos abertos.",
                    ValidationSeverity.ERROR,
                )
            )
    else:
        issues.append(
            _issue(
                "machine_profile_missing",
                "Nenhum perfil de máquina foi carregado; o MPR ficará bloqueado.",
                ValidationSeverity.WARNING,
            )
        )
    return issues


def has_critical_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == ValidationSeverity.ERROR for issue in issues)


def validate_machine_profile(profile: MachineProfile | None) -> list[ValidationIssue]:
    if profile is None:
        return [
            _issue(
                "mpr_profile_missing",
                "Carregue um perfil JSON de máquina.",
                ValidationSeverity.ERROR,
            )
        ]
    issues: list[ValidationIssue] = []
    if not profile.profile_name.strip():
        issues.append(
            _issue(
                "profile_name_missing",
                "O perfil precisa de um nome.",
                ValidationSeverity.ERROR,
            )
        )
    if profile.format.lower() != "mpr":
        issues.append(
            _issue(
                "profile_format",
                "O perfil carregado não declara o formato MPR.",
                ValidationSeverity.ERROR,
            )
        )
    if profile.units != "mm":
        issues.append(
            _issue(
                "profile_units",
                "Este MVP exporta MPR apenas quando o perfil declara milímetros.",
                ValidationSeverity.ERROR,
            )
        )
    if profile.decimal_separator not in {".", ","}:
        issues.append(
            _issue(
                "decimal_separator",
                "O separador decimal deve ser ponto ou vírgula.",
                ValidationSeverity.ERROR,
            )
        )
    if not 0 <= profile.coordinate_precision <= 6:
        issues.append(
            _issue(
                "coordinate_precision",
                "A precisão de coordenadas deve ficar entre 0 e 6.",
                ValidationSeverity.ERROR,
            )
        )
    for label, value in (
        ("largura máxima", profile.max_width_mm),
        ("altura máxima", profile.max_height_mm),
        ("profundidade máxima", profile.max_depth_mm),
    ):
        if value is not None and value <= 0:
            issues.append(
                _issue(
                    "profile_limit",
                    f"A {label} deve ser positiva quando informada.",
                    ValidationSeverity.ERROR,
                )
            )
    if profile.template_file:
        issues.append(
            _issue(
                "profile_template_path",
                "template_file não é acessado por segurança; cole o template na interface.",
                ValidationSeverity.ERROR,
            )
        )
    return issues

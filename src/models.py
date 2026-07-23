from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


Point = tuple[float, float]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} deve ser um número.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} deve ser um número finito.")
    return result


def _optional_float(value: Any, label: str) -> float | None:
    return None if value is None else _finite_float(value, label)


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} deve ser verdadeiro ou falso.")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} deve ser um número inteiro.")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} deve ser texto.")
    return value


def _point(value: Any, label: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} deve conter exatamente X e Y.")
    return _finite_float(value[0], f"{label}.X"), _finite_float(
        value[1], f"{label}.Y"
    )


class ValidationSeverity(str, Enum):
    ERROR = "erro"
    WARNING = "aviso"
    INFO = "informação"


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: ValidationSeverity
    contour_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationIssue":
        if not isinstance(data, dict):
            raise ValueError("Cada validação deve ser um objeto.")
        contour_id = data.get("contour_id")
        if contour_id is not None:
            contour_id = _strict_string(contour_id, "validation.contour_id")
        return cls(
            code=_strict_string(data.get("code"), "validation.code"),
            message=_strict_string(data.get("message"), "validation.message"),
            severity=ValidationSeverity(
                _strict_string(data.get("severity"), "validation.severity")
            ),
            contour_id=contour_id,
        )


@dataclass(slots=True)
class CNCContour:
    contour_id: str
    points_mm: list[Point]
    closed: bool
    is_hole: bool
    hierarchy_level: int
    selected: bool = True
    parent_id: str | None = None
    source_index: int | None = None
    compensation_mm: float = 0.0
    uncompensated_points_mm: list[Point] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points_mm"] = [[float(x), float(y)] for x, y in self.points_mm]
        data["uncompensated_points_mm"] = (
            [[float(x), float(y)] for x, y in self.uncompensated_points_mm]
            if self.uncompensated_points_mm is not None
            else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CNCContour":
        if not isinstance(data, dict):
            raise ValueError("Cada contorno deve ser um objeto JSON.")
        points = data.get("points_mm")
        if not isinstance(points, list):
            raise ValueError("points_mm deve ser uma lista.")
        base_points = data.get("uncompensated_points_mm")
        if base_points is not None and not isinstance(base_points, list):
            raise ValueError("uncompensated_points_mm deve ser uma lista ou nulo.")
        parent_id = data.get("parent_id")
        if parent_id is not None:
            parent_id = _strict_string(parent_id, "parent_id")
        source_index = data.get("source_index")
        if source_index is not None:
            source_index = _strict_int(source_index, "source_index")
        return cls(
            contour_id=_strict_string(data.get("contour_id"), "contour_id"),
            points_mm=[_point(point, "points_mm") for point in points],
            closed=_strict_bool(data.get("closed"), "closed"),
            is_hole=_strict_bool(data.get("is_hole"), "is_hole"),
            hierarchy_level=_strict_int(
                data.get("hierarchy_level"), "hierarchy_level"
            ),
            selected=_strict_bool(data.get("selected", True), "selected"),
            parent_id=parent_id,
            source_index=source_index,
            compensation_mm=_finite_float(
                data.get("compensation_mm", 0.0), "compensation_mm"
            ),
            uncompensated_points_mm=(
                [_point(point, "uncompensated_points_mm") for point in base_points]
                if base_points is not None
                else None
            ),
        )


@dataclass(slots=True)
class Calibration:
    method: str = ""
    pixels_per_mm: float | None = None
    reference_width_mm: float | None = None
    reference_height_mm: float | None = None
    point_1_px: Point | None = None
    point_2_px: Point | None = None
    reference_distance_mm: float | None = None

    @property
    def is_defined(self) -> bool:
        return self.pixels_per_mm is not None and self.pixels_per_mm > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        if not isinstance(data, dict):
            raise ValueError("calibration deve ser um objeto.")
        extra = set(data) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError("Campos desconhecidos na calibração: " + ", ".join(sorted(extra)))
        values: dict[str, Any] = {
            "method": _strict_string(data.get("method", ""), "calibration.method")
        }
        for key in (
            "pixels_per_mm",
            "reference_width_mm",
            "reference_height_mm",
            "reference_distance_mm",
        ):
            values[key] = _optional_float(data.get(key), f"calibration.{key}")
        for key in ("point_1_px", "point_2_px"):
            values[key] = (
                _point(data[key], f"calibration.{key}")
                if data.get(key) is not None
                else None
            )
        return cls(**values)


@dataclass(slots=True)
class ProcessingParameters:
    convert_grayscale: bool = True
    invert: bool = False
    gaussian_blur: bool = True
    gaussian_kernel: int = 5
    denoise: bool = False
    threshold_mode: str = "otsu"
    manual_threshold: int = 127
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    use_canny: bool = False
    canny_min: int = 50
    canny_max: int = 150
    morphology_kernel: int = 3
    closing_iterations: int = 0
    opening_iterations: int = 0
    dilation_iterations: int = 0
    erosion_iterations: int = 0
    min_area_mm2: float = 1.0
    min_perimeter_mm: float = 1.0
    ignore_small_noise: bool = True
    keep_largest_only: bool = False
    close_gaps: bool = True
    closing_tolerance_mm: float = 0.25
    simplify: bool = True
    simplify_tolerance_mm: float = 0.10
    smooth: bool = False
    smoothing_window: int = 3
    straighten_lines: bool = True
    straighten_auto_tolerance: bool = True
    straighten_tolerance_mm: float = 1.0
    preserve_rounded_sections: bool = True
    minimum_straight_length_mm: float = 40.0
    keep_holes: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingParameters":
        if not isinstance(data, dict):
            raise ValueError("processing deve ser um objeto.")
        extra = set(data) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(
                "Campos desconhecidos no processamento: " + ", ".join(sorted(extra))
            )
        defaults = cls()
        values: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            value = data.get(key, getattr(defaults, key))
            default = getattr(defaults, key)
            if isinstance(default, bool):
                values[key] = _strict_bool(value, f"processing.{key}")
            elif isinstance(default, int):
                values[key] = _strict_int(value, f"processing.{key}")
            elif isinstance(default, float):
                values[key] = _finite_float(value, f"processing.{key}")
            else:
                values[key] = _strict_string(value, f"processing.{key}")
        if values["threshold_mode"] not in {"manual", "otsu", "adaptativo"}:
            raise ValueError("processing.threshold_mode não é suportado.")
        return cls(**values)


@dataclass(slots=True)
class ToolSettings:
    diameter_mm: float = 6.0
    cutting_depth_mm: float = 3.0
    depth_per_pass_mm: float = 1.5
    feed_rate_mm_min: float = 1500.0
    plunge_rate_mm_min: float = 500.0
    spindle_rpm: int = 18000
    safety_margin_mm: float = 5.0
    cutting_direction: str = "convencional"
    units: str = "mm"

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSettings":
        if not isinstance(data, dict):
            raise ValueError("tool deve ser um objeto.")
        extra = set(data) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(
                "Campos desconhecidos na ferramenta: " + ", ".join(sorted(extra))
            )
        defaults = cls()
        values: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            value = data.get(key, getattr(defaults, key))
            default = getattr(defaults, key)
            if isinstance(default, int):
                values[key] = _strict_int(value, f"tool.{key}")
            elif isinstance(default, float):
                values[key] = _finite_float(value, f"tool.{key}")
            else:
                values[key] = _strict_string(value, f"tool.{key}")
        if values["cutting_direction"] not in {"convencional", "concordante"}:
            raise ValueError("tool.cutting_direction não é suportado.")
        if values["units"] != "mm":
            raise ValueError("tool.units deve ser mm.")
        return cls(**values)


@dataclass(slots=True)
class MachineProfile:
    profile_name: str = ""
    manufacturer: str = ""
    machine_model: str = ""
    controller: str = ""
    software: str = ""
    software_version: str = ""
    format: str = "mpr"
    units: str = "mm"
    decimal_separator: str = "."
    coordinate_precision: int = 3
    max_width_mm: float | None = None
    max_height_mm: float | None = None
    max_depth_mm: float | None = None
    allow_negative_coordinates: bool = False
    supports_open_contours: bool = False
    supports_arcs: bool = False
    supports_polylines: bool = True
    supports_tool_compensation: bool = False
    requires_clockwise_outer_contour: bool | None = None
    requires_clockwise_holes: bool | None = None
    template_file: str | None = None
    contour_block_template: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MachineProfile":
        if not isinstance(data, dict):
            raise ValueError("O perfil da máquina deve ser um objeto.")
        extra = set(data) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(
                "Campos desconhecidos no perfil: " + ", ".join(sorted(extra))
            )
        defaults = cls()
        values: dict[str, Any] = {}
        optional_numbers = {"max_width_mm", "max_height_mm", "max_depth_mm"}
        optional_booleans = {
            "requires_clockwise_outer_contour",
            "requires_clockwise_holes",
        }
        optional_strings = {"template_file", "contour_block_template"}
        for key in cls.__dataclass_fields__:
            value = data.get(key, getattr(defaults, key))
            default = getattr(defaults, key)
            if key in optional_numbers:
                values[key] = _optional_float(value, f"profile.{key}")
            elif key in optional_booleans:
                values[key] = (
                    None
                    if value is None
                    else _strict_bool(value, f"profile.{key}")
                )
            elif key in optional_strings:
                values[key] = (
                    None
                    if value is None
                    else _strict_string(value, f"profile.{key}")
                )
            elif isinstance(default, bool):
                values[key] = _strict_bool(value, f"profile.{key}")
            elif isinstance(default, int):
                values[key] = _strict_int(value, f"profile.{key}")
            else:
                values[key] = _strict_string(value, f"profile.{key}")
        return cls(**values)


@dataclass(slots=True)
class CNCProject:
    name: str = "novo_projeto"
    schema_version: str = "1.0"
    calibration: Calibration = field(default_factory=Calibration)
    processing: ProcessingParameters = field(default_factory=ProcessingParameters)
    contours: list[CNCContour] = field(default_factory=list)
    tool: ToolSettings = field(default_factory=ToolSettings)
    machine_profile: MachineProfile | None = None
    transformations: list[dict[str, Any]] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    image_metadata: dict[str, Any] = field(default_factory=dict)
    material_thickness_mm: float = 18.0

    @property
    def selected_contours(self) -> list[CNCContour]:
        return [contour for contour in self.contours if contour.selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "calibration": asdict(self.calibration),
            "processing": asdict(self.processing),
            "contours": [contour.to_dict() for contour in self.contours],
            "tool": asdict(self.tool),
            "machine_profile": (
                asdict(self.machine_profile) if self.machine_profile is not None else None
            ),
            "transformations": self.transformations,
            "validation_issues": [
                issue.to_dict() for issue in self.validation_issues
            ],
            "image_metadata": self.image_metadata,
            "material_thickness_mm": self.material_thickness_mm,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CNCProject":
        if not isinstance(data, dict):
            raise ValueError("O projeto deve ser um objeto JSON.")
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "O projeto contém valor não serializável ou número não finito."
            ) from exc
        if str(data.get("schema_version", "")) != "1.0":
            raise ValueError("Versão de projeto não suportada.")
        contours = data.get("contours", [])
        transformations = data.get("transformations", [])
        validation_issues = data.get("validation_issues", [])
        image_metadata = data.get("image_metadata", {})
        if not isinstance(contours, list):
            raise ValueError("contours deve ser uma lista.")
        if not isinstance(transformations, list) or not all(
            isinstance(item, dict) for item in transformations
        ):
            raise ValueError("transformations deve ser uma lista de objetos.")
        if not isinstance(validation_issues, list):
            raise ValueError("validation_issues deve ser uma lista.")
        if not isinstance(image_metadata, dict):
            raise ValueError("image_metadata deve ser um objeto.")
        return cls(
            name=_strict_string(data.get("name", "projeto_importado"), "name"),
            schema_version="1.0",
            calibration=Calibration.from_dict(data.get("calibration", {})),
            processing=ProcessingParameters.from_dict(data.get("processing", {})),
            contours=[CNCContour.from_dict(item) for item in contours],
            tool=ToolSettings.from_dict(data.get("tool", {})),
            machine_profile=(
                MachineProfile.from_dict(data["machine_profile"])
                if data.get("machine_profile")
                else None
            ),
            transformations=list(transformations),
            validation_issues=[
                ValidationIssue.from_dict(item) for item in validation_issues
            ],
            image_metadata=dict(image_metadata),
            material_thickness_mm=_finite_float(
                data.get("material_thickness_mm", 18.0),
                "material_thickness_mm",
            ),
        )

    @classmethod
    def from_json(cls, content: str) -> "CNCProject":
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("O JSON do projeto deve conter um objeto.")
        return cls.from_dict(parsed)

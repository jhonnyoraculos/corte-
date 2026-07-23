from __future__ import annotations

from abc import ABC, abstractmethod
import re

from ..geometry import dimensions
from ..models import (
    CNCContour,
    CNCProject,
    MachineProfile,
    ValidationIssue,
    ValidationSeverity,
)
from ..validators import validate_machine_profile
from .mpr_template import MPRReferenceAnalysis, validate_internal_template


PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class BaseMPRPostProcessor(ABC):
    @abstractmethod
    def validate_profile(self) -> list[ValidationIssue]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, project: CNCProject, contours: list[CNCContour]) -> str:
        raise NotImplementedError

    @abstractmethod
    def validate_output(self, output: str) -> list[ValidationIssue]:
        raise NotImplementedError


class TemplateMPRPostProcessor(BaseMPRPostProcessor):
    """Pós-processador que só usa sintaxe explicitamente fornecida pelo operador."""

    def __init__(
        self,
        profile: MachineProfile | None,
        reference: MPRReferenceAnalysis | None,
        program_template: str,
    ) -> None:
        self.profile = profile
        self.reference = reference
        self.program_template = program_template

    def validate_profile(self) -> list[ValidationIssue]:
        issues = validate_machine_profile(self.profile)
        if self.reference is None:
            issues.append(
                ValidationIssue(
                    "mpr_reference_missing",
                    "Carregue um arquivo MPR real de referência.",
                    ValidationSeverity.ERROR,
                )
            )
        elif self.reference.unsafe_external_paths:
            issues.append(
                ValidationIssue(
                    "mpr_external_path",
                    "O arquivo de referência contém caminho externo e foi bloqueado.",
                    ValidationSeverity.ERROR,
                )
            )
        for message in validate_internal_template(self.program_template):
            issues.append(
                ValidationIssue(
                    "mpr_program_template",
                    message,
                    ValidationSeverity.ERROR,
                )
            )
        block = self.profile.contour_block_template if self.profile else None
        if not block:
            issues.append(
                ValidationIssue(
                    "mpr_contour_template_missing",
                    "Defina contour_block_template no perfil da máquina.",
                    ValidationSeverity.ERROR,
                )
            )
        elif "{{X}}" not in block or "{{Y}}" not in block:
            issues.append(
                ValidationIssue(
                    "mpr_coordinate_template",
                    "O bloco de contorno precisa conter {{X}} e {{Y}}.",
                    ValidationSeverity.ERROR,
                )
            )
        return issues

    def _format_number(self, value: float) -> str:
        if self.profile is None:
            raise ValueError("Perfil da máquina ausente.")
        formatted = f"{value:.{self.profile.coordinate_precision}f}"
        if self.profile.decimal_separator == ",":
            formatted = formatted.replace(".", ",")
        return formatted

    def generate(self, project: CNCProject, contours: list[CNCContour]) -> str:
        profile_issues = self.validate_profile()
        if any(issue.severity == ValidationSeverity.ERROR for issue in profile_issues):
            raise ValueError(
                "Configuração MPR incompleta: "
                + "; ".join(issue.message for issue in profile_issues)
            )
        selected = [contour for contour in contours if contour.selected]
        if not selected:
            raise ValueError("Não há contornos selecionados para o MPR.")
        assert self.profile is not None
        assert self.reference is not None
        block_template = self.profile.contour_block_template or ""
        contour_blocks: list[str] = []
        for contour_index, contour in enumerate(selected, start=1):
            if not contour.closed and not self.profile.supports_open_contours:
                raise ValueError(
                    f"O perfil não permite o contorno aberto {contour.contour_id}."
                )
            point_blocks: list[str] = []
            for point_index, (x, y) in enumerate(contour.points_mm, start=1):
                replacements = {
                    "{{X}}": self._format_number(x),
                    "{{Y}}": self._format_number(y),
                    "{{POINT_INDEX}}": str(point_index),
                    "{{CONTOUR_INDEX}}": str(contour_index),
                    "{{CONTOUR_ID}}": contour.contour_id,
                    "{{IS_HOLE}}": "1" if contour.is_hole else "0",
                    "{{CLOSED}}": "1" if contour.closed else "0",
                }
                rendered = block_template
                for placeholder, value in replacements.items():
                    rendered = rendered.replace(placeholder, value)
                point_blocks.append(rendered)
            contour_blocks.append(self.reference.newline.join(point_blocks))

        width, height = dimensions(selected)
        replacements = {
            "{{PROJECT_NAME}}": project.name,
            "{{LENGTH}}": self._format_number(width),
            "{{WIDTH}}": self._format_number(height),
            "{{THICKNESS}}": self._format_number(project.material_thickness_mm),
            "{{CONTOUR_BLOCKS}}": self.reference.newline.join(contour_blocks),
        }
        output = self.program_template
        for placeholder, value in replacements.items():
            output = output.replace(placeholder, value)
        output_issues = self.validate_output(output)
        if any(issue.severity == ValidationSeverity.ERROR for issue in output_issues):
            raise ValueError(
                "Saída MPR rejeitada: "
                + "; ".join(issue.message for issue in output_issues)
            )
        return output

    def validate_output(self, output: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not output.strip():
            issues.append(
                ValidationIssue(
                    "mpr_empty_output",
                    "O pós-processador gerou saída vazia.",
                    ValidationSeverity.ERROR,
                )
            )
        unresolved = sorted(set(PLACEHOLDER_PATTERN.findall(output)))
        if unresolved:
            issues.append(
                ValidationIssue(
                    "mpr_unresolved_placeholders",
                    "Há placeholders sem valor: " + ", ".join(unresolved),
                    ValidationSeverity.ERROR,
                )
            )
        if re.search(r"(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|file://|https?://)", output):
            issues.append(
                ValidationIssue(
                    "mpr_output_external_path",
                    "A saída contém um caminho externo e foi bloqueada.",
                    ValidationSeverity.ERROR,
                )
            )
        return issues

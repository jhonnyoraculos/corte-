from __future__ import annotations

import pytest

from src.exporters.mpr_exporter import TemplateMPRPostProcessor
from src.exporters.mpr_template import analyze_mpr_reference
from src.models import MachineProfile, ValidationSeverity
from tests.helpers import valid_project


def profile() -> MachineProfile:
    return MachineProfile(
        profile_name="Perfil fictício somente para teste",
        contour_block_template="P{{POINT_INDEX}} X={{X}} Y={{Y}}",
        coordinate_precision=2,
        decimal_separator=",",
    )


def reference():
    return analyze_mpr_reference(b"HEADER\r\nREFERENCE CONTOUR\r\nEND\r\n")


def test_mpr_is_blocked_without_profile() -> None:
    processor = TemplateMPRPostProcessor(
        None, reference(), "HEADER\n{{CONTOUR_BLOCKS}}\nEND"
    )
    assert any(
        issue.severity == ValidationSeverity.ERROR
        for issue in processor.validate_profile()
    )


def test_mpr_is_blocked_without_reference() -> None:
    processor = TemplateMPRPostProcessor(
        profile(), None, "HEADER\n{{CONTOUR_BLOCKS}}\nEND"
    )
    assert any(
        issue.code == "mpr_reference_missing"
        for issue in processor.validate_profile()
    )


def test_template_generation_uses_only_configured_syntax() -> None:
    project = valid_project()
    template = (
        "NAME={{PROJECT_NAME}}\r\nL={{LENGTH}}\r\nW={{WIDTH}}\r\n"
        "{{CONTOUR_BLOCKS}}\r\nEND"
    )
    processor = TemplateMPRPostProcessor(profile(), reference(), template)
    output = processor.generate(project, project.contours)
    assert "NAME=teste" in output
    assert "L=100,00" in output
    assert "W=50,00" in output
    assert "P1 X=0,00 Y=0,00" in output
    assert "{{" not in output


@pytest.mark.parametrize(
    ("program_template", "block_template"),
    [
        ("HEADER\nEND", "X={{X}} Y={{Y}}"),
        ("HEADER\n{{CONTOUR_BLOCKS}}\nEND", "X={{X}}"),
    ],
)
def test_invalid_template_is_rejected(
    program_template: str, block_template: str
) -> None:
    invalid_profile = profile()
    invalid_profile.contour_block_template = block_template
    processor = TemplateMPRPostProcessor(
        invalid_profile, reference(), program_template
    )
    assert any(
        issue.severity == ValidationSeverity.ERROR
        for issue in processor.validate_profile()
    )
    with pytest.raises(ValueError):
        processor.generate(valid_project(), valid_project().contours)

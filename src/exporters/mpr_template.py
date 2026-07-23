from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

from ..utils.files import MAX_MPR_BYTES, UploadValidationError


UNSAFE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|file://|https?://)", re.IGNORECASE
)


@dataclass(slots=True)
class MPRReferenceAnalysis:
    text: str
    encoding: str
    newline: str
    decimal_separator: str | None
    repeated_lines: list[tuple[str, int]] = field(default_factory=list)
    unsafe_external_paths: list[str] = field(default_factory=list)


def analyze_mpr_reference(content: bytes) -> MPRReferenceAnalysis:
    if not content:
        raise UploadValidationError("O arquivo MPR de referência está vazio.")
    if len(content) > MAX_MPR_BYTES:
        raise UploadValidationError("O MPR de referência excede o limite de 2 MB.")
    if b"\x00" in content:
        raise UploadValidationError("O MPR parece binário e não será interpretado.")

    decoded: tuple[str, str] | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = (content.decode(encoding), encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UploadValidationError("Não foi possível identificar a codificação do MPR.")
    text, encoding = decoded
    if not text.strip():
        raise UploadValidationError("O MPR não contém texto utilizável.")
    controls = sum(
        ord(character) < 32 and character not in "\r\n\t" for character in text
    )
    if controls / max(len(text), 1) > 0.01:
        raise UploadValidationError("O MPR contém dados de controle não confiáveis.")

    newline_counts = {
        "\r\n": text.count("\r\n"),
        "\n": text.count("\n") - text.count("\r\n"),
        "\r": text.count("\r") - text.count("\r\n"),
    }
    newline = max(newline_counts, key=newline_counts.get)
    if newline_counts[newline] == 0:
        newline = "\n"
    separators = re.findall(r"\d([.,])\d", text)
    decimal_separator = Counter(separators).most_common(1)[0][0] if separators else None
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = [
        (line, count)
        for line, count in Counter(normalized_lines).most_common(20)
        if count > 1
    ]
    unsafe_paths = UNSAFE_PATH_PATTERN.findall(text)
    return MPRReferenceAnalysis(
        text=text,
        encoding=encoding,
        newline=newline,
        decimal_separator=decimal_separator,
        repeated_lines=repeated,
        unsafe_external_paths=unsafe_paths,
    )


def validate_internal_template(template: str) -> list[str]:
    errors: list[str] = []
    if not template.strip():
        errors.append("O modelo de programa está vazio.")
    if "{{CONTOUR_BLOCKS}}" not in template:
        errors.append("O modelo precisa do placeholder {{CONTOUR_BLOCKS}}.")
    return errors

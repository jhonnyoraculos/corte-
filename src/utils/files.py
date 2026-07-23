from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import io


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_MPR_BYTES = 2 * 1024 * 1024
MAX_PROJECT_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG", "BMP"}


class UploadValidationError(ValueError):
    """Falha compreensível de validação de upload."""


def safe_filename(filename: str, default: str = "arquivo") -> str:
    raw = Path(filename or default).name
    stem = unicodedata.normalize("NFKD", Path(raw).stem)
    stem = stem.encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._-") or default
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(raw).suffix.lower())
    return f"{stem[:80]}{suffix[:10]}"


def validate_image_upload(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise UploadValidationError("Extensão de imagem não permitida.")
    if not content:
        raise UploadValidationError("O arquivo de imagem está vazio.")
    if len(content) > MAX_IMAGE_BYTES:
        raise UploadValidationError("A imagem excede o limite de 20 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            detected_format = image.format
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadValidationError("O conteúdo não é uma imagem válida.") from exc
    if detected_format not in ALLOWED_PIL_FORMATS:
        raise UploadValidationError("O formato real da imagem não é permitido.")
    expected = "JPEG" if suffix in {".jpg", ".jpeg"} else suffix[1:].upper()
    if detected_format != expected:
        raise UploadValidationError(
            "A extensão não corresponde ao formato real da imagem."
        )
    if width <= 0 or height <= 0:
        raise UploadValidationError("A imagem possui dimensões inválidas.")
    return {
        "filename": safe_filename(filename),
        "format": detected_format,
        "width_px": width,
        "height_px": height,
        "size_bytes": len(content),
    }


def load_json_upload(content: bytes, max_bytes: int = MAX_PROJECT_BYTES) -> dict[str, Any]:
    if not content or len(content) > max_bytes:
        raise UploadValidationError("Arquivo JSON vazio ou acima do limite permitido.")
    try:
        parsed = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadValidationError("JSON inválido ou com codificação não suportada.") from exc
    if not isinstance(parsed, dict):
        raise UploadValidationError("O JSON deve conter um objeto na raiz.")
    return parsed

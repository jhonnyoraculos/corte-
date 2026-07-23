from __future__ import annotations

from dataclasses import dataclass
import io
import logging

import cv2
import numpy as np
from PIL import Image

from .models import ProcessingParameters

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageProcessingResult:
    original_rgb: np.ndarray
    grayscale: np.ndarray
    binary: np.ndarray
    edges: np.ndarray
    contour_source: np.ndarray
    threshold_value: float | None


def decode_image(content: bytes) -> np.ndarray:
    """Decodifica bytes já validados para RGB, sem executar conteúdo enviado."""
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"))


def image_quality_warnings(image_rgb: np.ndarray) -> list[str]:
    warnings: list[str] = []
    height, width = image_rgb.shape[:2]
    if min(width, height) < 300 or width * height < 200_000:
        warnings.append(
            "A imagem tem baixa resolução; detalhes pequenos podem não ser detectados."
        )
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if focus_score < 50.0:
        warnings.append(
            f"A imagem pode estar desfocada (índice de nitidez {focus_score:.1f})."
        )
    return warnings


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def process_image(
    image_rgb: np.ndarray, params: ProcessingParameters
) -> ImageProcessingResult:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("A imagem de entrada deve estar no formato RGB.")

    gray = (
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if params.convert_grayscale
        else np.max(image_rgb, axis=2).astype(np.uint8)
    )
    working = gray.copy()
    if params.invert:
        working = cv2.bitwise_not(working)
    if params.gaussian_blur:
        kernel = _odd(params.gaussian_kernel)
        working = cv2.GaussianBlur(working, (kernel, kernel), 0)
    if params.denoise:
        working = cv2.medianBlur(working, 3)

    threshold_value: float | None = None
    if params.threshold_mode == "manual":
        threshold_value, binary = cv2.threshold(
            working, int(params.manual_threshold), 255, cv2.THRESH_BINARY
        )
    elif params.threshold_mode == "adaptativo":
        block_size = _odd(params.adaptive_block_size)
        binary = cv2.adaptiveThreshold(
            working,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(params.adaptive_c),
        )
    else:
        threshold_value, binary = cv2.threshold(
            working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    kernel_size = max(1, int(params.morphology_kernel))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    if params.closing_iterations:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(params.closing_iterations),
        )
    if params.opening_iterations:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=int(params.opening_iterations),
        )
    if params.dilation_iterations:
        binary = cv2.dilate(
            binary, kernel, iterations=int(params.dilation_iterations)
        )
    if params.erosion_iterations:
        binary = cv2.erode(
            binary, kernel, iterations=int(params.erosion_iterations)
        )

    edges = cv2.Canny(
        working, int(params.canny_min), int(params.canny_max)
    )
    contour_source = edges if params.use_canny else binary
    logger.info(
        "Imagem processada: modo=%s, limiar=%s, Canny=%s",
        params.threshold_mode,
        threshold_value,
        params.use_canny,
    )
    return ImageProcessingResult(
        original_rgb=image_rgb,
        grayscale=gray,
        binary=binary,
        edges=edges,
        contour_source=contour_source,
        threshold_value=threshold_value,
    )

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.contour_processing import extract_contours
from src.image_processing import image_quality_warnings, process_image
from src.models import ProcessingParameters
from src.utils.files import UploadValidationError, validate_image_upload


def extraction_params() -> ProcessingParameters:
    return ProcessingParameters(
        min_area_mm2=0,
        min_perimeter_mm=0,
        ignore_small_noise=False,
        simplify=False,
    )


def test_external_and_hole_hierarchy() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (180, 180), 255, thickness=-1)
    cv2.circle(image, (100, 100), 35, 0, thickness=-1)
    contours = extract_contours(image, 1.0, extraction_params())
    assert len(contours) == 2
    outer = next(contour for contour in contours if not contour.is_hole)
    hole = next(contour for contour in contours if contour.is_hole)
    assert outer.hierarchy_level == 0
    assert hole.hierarchy_level == 1
    assert hole.parent_id == outer.contour_id


def test_two_separate_parts_are_external() -> None:
    image = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (100, 150), 255, thickness=-1)
    cv2.rectangle(image, (180, 30), (280, 170), 255, thickness=-1)
    contours = extract_contours(image, 1.0, extraction_params())
    assert len(contours) == 2
    assert all(not contour.is_hole for contour in contours)


def test_extraction_straightens_pixelated_diagonal_automatically() -> None:
    image = np.zeros((220, 220), dtype=np.uint8)
    cv2.fillPoly(
        image,
        [np.array([(20, 200), (200, 20), (200, 200)], dtype=np.int32)],
        255,
    )
    raw_params = extraction_params()
    raw_params.straighten_lines = False
    raw = extract_contours(image, 1.0, raw_params)

    straight_params = extraction_params()
    straight_params.straighten_lines = True
    straight_params.straighten_auto_tolerance = True
    straight = extract_contours(image, 1.0, straight_params)

    assert len(raw) == len(straight) == 1
    assert len(straight[0].points_mm) == 3
    assert len(straight[0].points_mm) < len(raw[0].points_mm)


def test_synthetic_images_process_without_external_files() -> None:
    images: list[np.ndarray] = []
    rectangle = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.rectangle(rectangle, (20, 20), (130, 120), (0, 0, 0), -1)
    images.append(rectangle)
    circle = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.circle(circle, (75, 75), 50, (0, 0, 0), -1)
    images.append(circle)
    open_contour = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.polylines(
        open_contour,
        [np.array([(20, 120), (20, 20), (120, 20)])],
        False,
        (0, 0, 0),
        3,
    )
    images.append(open_contour)
    noisy = rectangle.copy()
    rng = np.random.default_rng(42)
    indexes = rng.integers(0, 150, size=(200, 2))
    noisy[indexes[:, 0], indexes[:, 1]] = rng.integers(
        0, 256, size=(200, 3), dtype=np.uint8
    )
    images.append(noisy)
    for image in images:
        result = process_image(image, ProcessingParameters())
        assert result.binary.shape == image.shape[:2]
        assert result.edges.dtype == np.uint8


def test_uploaded_image_quality_check_does_not_depend_on_processing_params() -> None:
    image = np.full((320, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (540, 240), (0, 0, 0), thickness=-1)
    warnings = image_quality_warnings(image)
    assert isinstance(warnings, list)


def test_processing_can_use_brightest_channel_when_grayscale_is_disabled() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    result = process_image(
        image,
        ProcessingParameters(
            convert_grayscale=False,
            gaussian_blur=False,
            threshold_mode="manual",
            manual_threshold=100,
        ),
    )
    assert np.all(result.grayscale == 200)


def test_malformed_image_upload_is_rejected() -> None:
    with pytest.raises(UploadValidationError):
        validate_image_upload("falsa.png", b"isto nao e uma imagem")
    with pytest.raises(UploadValidationError):
        validate_image_upload("imagem.exe", b"qualquer conteudo")

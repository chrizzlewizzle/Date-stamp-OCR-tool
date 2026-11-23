from __future__ import annotations

import cv2
import numpy as np

from datestamp.detect import detect_stamp, expected_stamp_corners


def _empty_cal() -> dict:
    # No calibration override: exercise the built-in orientation policy.
    return {"hsv_ranges": []}


def _draw_digit_row(img: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a row of digit-like vertical bars."""
    for i in range(6):
        x0 = x + i * 45
        cv2.rectangle(img, (x0, y), (x0 + 30, y + 80), color, -1)


def _landscape_with_stamp() -> np.ndarray:
    img = np.full((600, 800, 3), 245, dtype=np.uint8)
    # Distractor in the top-left to verify that the expected corner wins.
    cv2.rectangle(img, (40, 40), (260, 150), (0, 0, 220), -1)
    _draw_digit_row(img, 500, 500, (0, 128, 255))  # orange in BGR
    return img


def test_expected_corners_by_orientation() -> None:
    assert expected_stamp_corners(600, 800, _empty_cal()) == ["bottom_right"]
    assert expected_stamp_corners(800, 600, _empty_cal()) == ["bottom_left", "top_right"]
    # Nearly square frames are ambiguous.
    assert set(expected_stamp_corners(800, 800, _empty_cal())) == {
        "bottom_right",
        "bottom_left",
        "top_right",
        "top_left",
    }


def test_landscape_detects_bottom_right_only() -> None:
    img = _landscape_with_stamp()
    det, crop = detect_stamp(img, cal=_empty_cal())
    assert det.found
    assert det.corner == "bottom_right"
    assert crop is not None and crop.size > 0


def test_portrait_rotated_clockwise_detects_bottom_left_and_uprights_crop() -> None:
    # A landscape frame rotated clockwise puts its bottom-right stamp in the
    # portrait image's bottom-left corner.
    landscape = _landscape_with_stamp()
    portrait = cv2.rotate(landscape, cv2.ROTATE_90_CLOCKWISE)

    det, crop = detect_stamp(portrait, cal=_empty_cal())
    assert det.found
    assert det.corner == "bottom_left"
    assert crop is not None
    # The returned crop must be upright (wide), not a rotated tall strip.
    assert crop.shape[1] > crop.shape[0]
    assert any("crop_rotated_upright" in n for n in det.notes)


def test_portrait_rotated_counterclockwise_detects_top_right_and_uprights_crop() -> None:
    # A landscape frame rotated counter-clockwise puts its bottom-right stamp in
    # the portrait image's top-right corner.
    landscape = _landscape_with_stamp()
    portrait = cv2.rotate(landscape, cv2.ROTATE_90_COUNTERCLOCKWISE)

    det, crop = detect_stamp(portrait, cal=_empty_cal())
    assert det.found
    assert det.corner == "top_right"
    assert crop is not None
    assert crop.shape[1] > crop.shape[0]
    assert any("crop_rotated_upright" in n for n in det.notes)


def test_faint_overexposed_stamp_uses_fallback_ranges() -> None:
    img = np.full((600, 800, 3), 250, dtype=np.uint8)
    # Very bright, low-saturation pink: outside the calibrated S/V ranges.
    _draw_digit_row(img, 500, 500, (195, 215, 255))

    det, crop = detect_stamp(img, cal=_empty_cal())
    assert det.found
    assert det.corner == "bottom_right"
    assert any("faint_stamp_fallback_ranges" in n for n in det.notes)
    assert crop is not None


def test_blank_image_has_no_stamp() -> None:
    img = np.full((600, 800, 3), 250, dtype=np.uint8)
    det, crop = detect_stamp(img, cal=_empty_cal())
    assert not det.found
    assert crop is None

import io

import pytest
from PIL import Image

from preprocessing import InvalidImageError, load_and_validate, normalize, prepare_image_bytes

MAX_DIMENSION = 1024


def _decode(jpeg_bytes):
    return Image.open(io.BytesIO(jpeg_bytes))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_return_reencoded_jpeg(image_bytes_factory, fmt):
    raw = image_bytes_factory(fmt=fmt)

    out = prepare_image_bytes(raw)

    decoded = _decode(out)
    assert decoded.format == "JPEG"


def test_corrupt_bytes_raise_invalid_image_error(corrupt_image_bytes):
    with pytest.raises(InvalidImageError):
        prepare_image_bytes(corrupt_image_bytes)


@pytest.mark.parametrize("fmt", ["BMP", "GIF"])
def test_unsupported_formats_raise_invalid_image_error(image_bytes_factory, fmt):
    raw = image_bytes_factory(fmt=fmt)

    with pytest.raises(InvalidImageError):
        prepare_image_bytes(raw)


def test_non_square_image_becomes_square(image_bytes_factory):
    raw = image_bytes_factory(size=(400, 200))

    out = prepare_image_bytes(raw)

    decoded = _decode(out)
    assert decoded.width == decoded.height


def test_oversized_image_resized_down_to_max_dimension(image_bytes_factory):
    raw = image_bytes_factory(size=(1500, 1500))

    out = prepare_image_bytes(raw)

    decoded = _decode(out)
    assert decoded.size == (MAX_DIMENSION, MAX_DIMENSION)


def test_oversized_non_square_image_cropped_then_resized(image_bytes_factory):
    raw = image_bytes_factory(size=(2000, 1500))

    out = prepare_image_bytes(raw)

    decoded = _decode(out)
    assert decoded.size == (MAX_DIMENSION, MAX_DIMENSION)


def test_exif_rotation_is_corrected():
    # Square image (so the center-crop step is a no-op and doesn't disturb the
    # marker) with a red 60x60 marker in the top-left corner on a blue field.
    img = Image.new("RGB", (300, 300), (0, 0, 255))
    for x in range(60):
        for y in range(60):
            img.putpixel((x, y), (255, 0, 0))

    # Orientation=3 means "stored data needs a 180-degree rotation to display
    # correctly" — a rotation that unambiguously moves the top-left marker to
    # the bottom-right corner, regardless of directional (CW/CCW) convention.
    exif = img.getexif()
    exif[0x0112] = 3
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    loaded = load_and_validate(buf.getvalue())
    normalized = normalize(loaded)

    w, h = normalized.size
    top_left = normalized.getpixel((5, 5))
    bottom_right = normalized.getpixel((w - 5, h - 5))

    assert top_left[2] > 200 and top_left[0] < 50  # now blue (background)
    assert bottom_right[0] > 200 and bottom_right[2] < 50  # now red (marker)

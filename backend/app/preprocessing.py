"""
Image preprocessing.

Jewellery visual search lives or dies on image normalization: a customer's
WhatsApp photo (cluttered background, on a hand, poor lighting) embedded
directly against clean studio catalog shots produces unreliable similarity
scores. This module standardizes any incoming image before it reaches the
embedding model.
"""
import io
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_DIMENSION = 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(Exception):
    pass


def load_and_validate(image_bytes: bytes) -> Image.Image:
    """Load raw bytes into a PIL Image, raising a clear error on bad input."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except UnidentifiedImageError as exc:
        raise InvalidImageError("File is not a readable image (jpg/png/webp).") from exc

    if img.format not in ALLOWED_FORMATS:
        raise InvalidImageError(f"Unsupported format '{img.format}'. Use JPEG, PNG, or WEBP.")

    return img


def normalize(img: Image.Image) -> Image.Image:
    """
    Standardize an image before embedding:
    - correct EXIF rotation
    - convert to RGB (strip alpha/CMYK inconsistencies)
    - center-crop to a square aspect ratio (jewellery product shots are ~square)
    - resize to a consistent max dimension to keep embedding cost/latency predictable
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    if side > MAX_DIMENSION:
        img = img.resize((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    return img


def prepare_image_bytes(raw_bytes: bytes) -> bytes:
    """Full pipeline: validate, normalize, re-encode to JPEG bytes."""
    img = load_and_validate(raw_bytes)
    img = normalize(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

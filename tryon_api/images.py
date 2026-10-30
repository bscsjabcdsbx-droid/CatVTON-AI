"""Kiểm tra và giải mã ảnh hoàn toàn trong RAM (không ghi file tạm)."""
from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    """Ảnh không hợp lệ (sai định dạng, hỏng, quá lớn)."""


def sniff_format(data: bytes) -> str | None:
    """Nhận diện định dạng thật bằng magic bytes, không tin Content-Type."""
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def decode_rgb(data: bytes, max_pixels: int) -> Image.Image:
    """Giải mã bytes -> ảnh RGB đã xoay đúng theo EXIF. Ném InvalidImage nếu không hợp lệ."""
    if not data:
        raise InvalidImage("Ảnh rỗng")
    fmt = sniff_format(data)
    if fmt is None:
        raise InvalidImage("Chỉ chấp nhận JPEG, PNG hoặc WebP")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            if img.format not in ALLOWED_FORMATS or img.format != fmt:
                raise InvalidImage("Định dạng ảnh không khớp nội dung")
            w, h = img.size
            if w < 64 or h < 64:
                raise InvalidImage("Ảnh quá nhỏ")
            if w * h > max_pixels:
                raise InvalidImage("Ảnh có độ phân giải quá lớn")
            img.load()
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Không đọc được ảnh") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()

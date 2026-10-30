"""Cấu hình CatVTON API, đọc từ biến môi trường (và file .env nếu có).

Không có giá trị bí mật nào được ghi cứng ở đây.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Đọc file .env đơn giản (KEY=VALUE). Biến đã có trong môi trường được giữ nguyên."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} phải là số nguyên, nhận được {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} phải >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} phải <= {maximum}")
    return value


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} phải là số thực, nhận được {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Mạng & bảo mật
    host: str
    port: int
    api_key: str
    # Hàng đợi & giới hạn
    max_concurrent_inference: int
    max_queue_size: int
    max_job_timeout: int          # giây, tính từ lúc bắt đầu xử lý
    result_ttl: int               # giây giữ ảnh kết quả trong RAM nếu chưa ai lấy
    result_delivered_grace: int   # giây giữ thêm sau khi Shop_API đã lấy ảnh lần đầu
    job_retention: int            # giây giữ metadata của job đã kết thúc
    max_stored_results: int       # số ảnh kết quả tối đa giữ đồng thời trong RAM
    upload_max_size: int          # byte, mỗi ảnh
    max_image_pixels: int
    # Model
    base_model: str
    attn_repo: str
    mixed_precision: str
    width: int
    height: int
    default_steps: int
    min_steps: int
    max_steps: int
    default_guidance: float
    masker_device: str
    vae_tiling: bool
    skip_safety_check: bool
    default_seed: int


def load_settings(env_file: Path | None = None) -> Settings:
    if env_file is not None:
        load_dotenv(env_file)

    width = _int("TRYON_WIDTH", 768, 256, 1536)
    height = _int("TRYON_HEIGHT", 1024, 256, 2048)
    if width % 8 or height % 8:
        raise ValueError("TRYON_WIDTH và TRYON_HEIGHT phải chia hết cho 8")

    precision = os.environ.get("CATVTON_MIXED_PRECISION", "bf16").strip() or "bf16"
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("CATVTON_MIXED_PRECISION phải là no, fp16 hoặc bf16")

    masker_device = os.environ.get("TRYON_MASKER_DEVICE", "cuda").strip() or "cuda"
    if masker_device not in {"cuda", "cpu"}:
        raise ValueError("TRYON_MASKER_DEVICE phải là cuda hoặc cpu")

    min_steps = _int("TRYON_MIN_STEPS", 10, 1, 100)
    max_steps = _int("TRYON_MAX_STEPS", 50, min_steps, 100)
    default_steps = _int("TRYON_DEFAULT_STEPS", 30, min_steps, max_steps)

    return Settings(
        host=os.environ.get("CATVTON_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_int("CATVTON_PORT", 8000, 1, 65535),
        api_key=os.environ.get("CATVTON_API_KEY", "").strip(),
        max_concurrent_inference=_int("MAX_CONCURRENT_INFERENCE", 1, 1, 4),
        max_queue_size=_int("TRYON_MAX_QUEUE_SIZE", 5, 0, 100),
        max_job_timeout=_int("MAX_JOB_TIMEOUT", 180, 10, 3600),
        result_ttl=_int("TRYON_RESULT_TTL", 3600, 30, 86400),
        result_delivered_grace=_int("TRYON_RESULT_DELIVERED_GRACE", 300, 10, 86400),
        job_retention=_int("TRYON_JOB_RETENTION", 3600, 60, 86400),
        max_stored_results=_int("TRYON_MAX_STORED_RESULTS", 20, 1, 500),
        upload_max_size=_int("TRYON_UPLOAD_MAX_SIZE", 10 * 1024 * 1024, 1024, 50 * 1024 * 1024),
        max_image_pixels=_int("TRYON_MAX_IMAGE_PIXELS", 50_000_000, 10_000, 200_000_000),
        base_model=os.environ.get("CATVTON_BASE_MODEL", "booksforcharlie/stable-diffusion-inpainting").strip(),
        attn_repo=os.environ.get("CATVTON_ATTN_REPO", "zhengchong/CatVTON").strip(),
        mixed_precision=precision,
        width=width,
        height=height,
        default_steps=default_steps,
        min_steps=min_steps,
        max_steps=max_steps,
        default_guidance=_float("TRYON_DEFAULT_GUIDANCE", 2.5),
        masker_device=masker_device,
        vae_tiling=_bool("TRYON_VAE_TILING", False),
        skip_safety_check=_bool("TRYON_SKIP_SAFETY_CHECK", False),
        default_seed=_int("TRYON_DEFAULT_SEED", 42, -1, 2**31 - 1),
    )

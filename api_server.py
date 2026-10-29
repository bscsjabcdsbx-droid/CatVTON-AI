
import io
import os
import uuid
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


# --------------------------------------------------
# 1. Cấu hình
# --------------------------------------------------

BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"
MODEL_REPO = "zhengchong/CatVTON"

WIDTH = 768
HEIGHT = 1024

OUTPUT_DIR = Path("resource/demo/output/api")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CatVTON API", version="1.0.0")


# --------------------------------------------------
# 2. Khởi tạo model MỘT LẦN khi khởi động server
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "Không tìm thấy CUDA. Hãy kiểm tra PyTorch và driver GPU."
    )

device = "cuda"

print("Đang tải cấu hình CatVTON...")
repo_path = snapshot_download(repo_id=MODEL_REPO)

print("Đang khởi tạo pipeline...")
pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device=device,
)

print("Đang khởi tạo bộ tạo mask...")
mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device,
)

print("CatVTON API đã sẵn sàng.")


# --------------------------------------------------
# 3. Kiểm tra server
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
    }


# --------------------------------------------------
# 4. API tạo ảnh thử đồ
# --------------------------------------------------

@app.post("/try-on")
def try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = Form("upper"),
):
    if cloth_type not in {"upper", "lower", "overall"}:
        raise HTTPException(
            status_code=400,
            detail="cloth_type phải là upper, lower hoặc overall",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    for upload in (person_image, cloth_image):
        if upload.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Chỉ chấp nhận ảnh JPEG, PNG hoặc WEBP",
            )

    try:
        person_bytes = person_image.file.read()
        cloth_bytes = cloth_image.file.read()

        # Giới hạn kích thước file: 8 MB mỗi ảnh
        if len(person_bytes) > 8 * 1024 * 1024:
            raise HTTPException(400, "Ảnh người vượt quá 8 MB")

        if len(cloth_bytes) > 8 * 1024 * 1024:
            raise HTTPException(400, "Ảnh quần áo vượt quá 8 MB")

        person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        cloth = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Không đọc được ảnh. Hãy chọn một file ảnh hợp lệ.",
        )

    # Chuẩn hóa kích thước như ứng dụng CatVTON gốc
    person = resize_and_crop(person, (WIDTH, HEIGHT))
    cloth = resize_and_padding(cloth, (WIDTH, HEIGHT))

    # Tự tạo mask cho vùng quần áo cần thay
    mask = automasker(person, cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"

    try:
        # Không cho tạo gradient để tiết kiệm bộ nhớ
        with torch.inference_mode():
            result = pipeline(
                image=person,
                condition_image=cloth,
                mask=mask,
                num_inference_steps=30,
                guidance_scale=2.5,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )[0]

        result.save(output_path, format="PNG")

    except Exception as exc:
        if output_path.exists():
            output_path.unlink()

        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=503,
                detail="GPU hết bộ nhớ. Hãy giảm độ phân giải hoặc đóng ứng dụng đang dùng GPU.",
            )

        print("Lỗi tạo ảnh CatVTON:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="CatVTON không tạo được ảnh. Kiểm tra log của server.",
        )

    finally:
        person_image.file.close()
        cloth_image.file.close()

    # Gửi PNG về Express, rồi xóa file tạm sau khi gửi xong
    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename="try-on.png",
        background=BackgroundTask(
            lambda: output_path.unlink(missing_ok=True)
        ),
    )

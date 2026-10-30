"""Bọc CatVTONPipeline + AutoMasker có sẵn của repo. Model chỉ được tải một lần.

Không viết lại thuật toán: dùng đúng các bước của app.py / api_server.py gốc
(resize_and_crop, resize_and_padding, AutoMasker, mask blur, CatVTONPipeline).
"""
from __future__ import annotations

import gc
import logging
import os
import random
from typing import Callable

from PIL import Image

from .images import encode_png
from .job_queue import EngineError, JobCancelled, JobTimedOut
from .settings import Settings

log = logging.getLogger("catvton.engine")


class CatVTONEngine:
    def __init__(self, settings: Settings) -> None:
        import torch
        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download

        from model.cloth_masker import AutoMasker
        from model.pipeline import CatVTONPipeline
        from utils import init_weight_dtype, resize_and_crop, resize_and_padding

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")

        self._torch = torch
        self._resize_and_crop = resize_and_crop
        self._resize_and_padding = resize_and_padding
        self.settings = settings
        self.device = "cuda"

        log.info("Tải checkpoint CatVTON (%s)...", settings.attn_repo)
        repo_path = snapshot_download(repo_id=settings.attn_repo)

        log.info("Khởi tạo pipeline (%s, %s)...", settings.base_model, settings.mixed_precision)
        self.pipeline = CatVTONPipeline(
            base_ckpt=settings.base_model,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype(settings.mixed_precision),
            use_tf32=True,
            device=self.device,
            skip_safety_check=settings.skip_safety_check,
        )
        if settings.vae_tiling:
            # Giảm VRAM đỉnh khi encode/decode VAE (đổi lại chậm hơn một chút).
            self.pipeline.vae.enable_tiling()

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        log.info("Khởi tạo AutoMasker trên %s...", settings.masker_device)
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device=settings.masker_device,
        )
        self._free_memory()
        log.info("CatVTON engine sẵn sàng.")

    # ------------------------------------------------------------------
    def gpu_info(self) -> dict:
        torch = self._torch
        try:
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "name": props.name,
                "memory_total_mb": round(props.total_memory / 2**20),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 2**20),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 2**20),
            }
        except Exception:
            return {"available": False}

    def _free_memory(self) -> None:
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def encode(self, image: Image.Image) -> bytes:
        return encode_png(image)

    def run(self, person: Image.Image, cloth: Image.Image, cloth_type: str, steps: int,
            guidance: float, seed: int, on_step: Callable[[int, int], None]) -> Image.Image:
        torch = self._torch
        s = self.settings
        size = (s.width, s.height)
        try:
            person_r = self._resize_and_crop(person, size)
            cloth_r = self._resize_and_padding(cloth, size)
            with torch.inference_mode():
                mask = self.automasker(person_r, cloth_type)["mask"]
                mask = self.mask_processor.blur(mask, blur_factor=9)
                on_step(0, steps)  # kiểm tra hủy/timeout sau bước tạo mask
                if seed < 0:
                    seed = random.randint(0, 2**31 - 1)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                images, is_nsfw = self.pipeline(
                    image=person_r,
                    condition_image=cloth_r,
                    mask=mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=s.height,
                    width=s.width,
                    generator=generator,
                    step_callback=on_step,
                    return_nsfw_status=True,
                )

            if is_nsfw:
                raise EngineError("NSFW_BLOCKED")

            result = images[0]
            del images
            return result
        except (JobCancelled, JobTimedOut):
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise EngineError("GPU_OOM") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineError("GPU_OOM") from exc
            raise
        finally:
            # Giải phóng tensor trung gian và VRAM cache sau MỖI job, kể cả khi lỗi.
            person_r = cloth_r = mask = generator = None  # noqa: F841
            self._free_memory()

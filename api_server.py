"""CatVTON API (FastAPI) cho Shop_API.

Chạy:  python api_server.py           (đọc cấu hình từ .env, xem .env.example)

Endpoint (tất cả trừ /livez yêu cầu header X-API-Key):
  GET    /livez                 kiểm tra tiến trình còn sống (không lộ thông tin)
  GET    /health                trạng thái service, model, GPU, hàng đợi
  POST   /try-on                tạo tác vụ (multipart): person_image, cloth_image, cloth_type,
                                job_id?, num_inference_steps?, guidance_scale?, seed?  -> 202
  GET    /jobs/{job_id}         trạng thái tác vụ
  GET    /jobs/{job_id}/result  ảnh PNG kết quả (bytes trực tiếp)
  DELETE /jobs/{job_id}         hủy tác vụ đang chờ / yêu cầu dừng tác vụ đang chạy / giải phóng ảnh kết quả

Giao diện Gradio (app.py) không bị thay đổi và vẫn chạy độc lập như trước.
"""
from __future__ import annotations

import hmac
import logging
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from tryon_api.images import InvalidImage, decode_rgb
from tryon_api.job_queue import CLOTH_TYPES, JobQueue, NotReady, QueueFull
from tryon_api.settings import Settings, load_settings

log = logging.getLogger("catvton.api")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _err(status: int, code: str, detail: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse({"error_code": code, "detail": detail}, status_code=status, headers=headers)


def _keep_uploads_in_memory(limit: int) -> None:
    """Starlette mặc định ghi file upload > 1MB ra file tạm trên đĩa. Nâng ngưỡng để ảnh chỉ nằm trong RAM."""
    try:
        from starlette.formparsers import MultiPartParser
        if hasattr(MultiPartParser, "spool_max_size"):
            MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, limit + 1024 * 1024)
        else:
            log.warning("Phiên bản Starlette không hỗ trợ spool_max_size; ảnh lớn có thể được đệm ra file tạm.")
    except Exception:  # pragma: no cover
        log.warning("Không cấu hình được bộ đệm multipart.")


def create_app(settings: Settings, engine_factory: Callable[[Settings], object]) -> FastAPI:
    if not settings.api_key or len(settings.api_key) < 24:
        raise RuntimeError("CATVTON_API_KEY phải được đặt và dài ít nhất 24 ký tự (xem .env.example)")

    _keep_uploads_in_memory(settings.upload_max_size)
    state: dict = {"engine": None, "status": "loading", "error": None}

    queue = JobQueue(
        lambda: state["engine"],
        max_concurrent=settings.max_concurrent_inference,
        max_queue_size=settings.max_queue_size,
        job_timeout=settings.max_job_timeout,
        result_ttl=settings.result_ttl,
        delivered_grace=settings.result_delivered_grace,
        job_retention=settings.job_retention,
        max_stored_results=settings.max_stored_results,
    )

    def load_model() -> None:
        try:
            state["engine"] = engine_factory(settings)
            state["status"] = "ready"
        except Exception as exc:
            msg = str(exc)
            state["status"] = "unavailable"
            state["error"] = "CUDA_UNAVAILABLE" if "CUDA_UNAVAILABLE" in msg else "MODEL_LOAD_FAILED"
            log.error("Không tải được model: %s", type(exc).__name__)
            log.debug("Chi tiết", exc_info=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        queue.start()
        threading.Thread(target=load_model, name="model-loader", daemon=True).start()
        yield
        queue.stop()

    app = FastAPI(title="CatVTON API", version="2.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.queue = queue
    app.state.model_state = state

    max_body = settings.upload_max_size * 2 + 256 * 1024

    @app.middleware("http")
    async def auth_and_size(request: Request, call_next):
        if request.url.path != "/livez":
            provided = request.headers.get("x-api-key", "")
            if not provided:
                auth = request.headers.get("authorization", "")
                provided = auth[7:] if auth.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(provided.encode(), settings.api_key.encode()):
                return _err(401, "UNAUTHORIZED", "Thiếu hoặc sai API key")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > max_body:
                    return _err(413, "PAYLOAD_TOO_LARGE", "Dữ liệu gửi lên quá lớn")
            except ValueError:
                return _err(400, "BAD_REQUEST", "Content-Length không hợp lệ")
        return await call_next(request)

    @app.get("/livez")
    def livez():
        return {"ok": True}

    @app.get("/health")
    def health():
        engine = state["engine"]
        gpu = engine.gpu_info() if engine is not None and hasattr(engine, "gpu_info") else {"available": False}
        stats = queue.stats()
        if state["status"] == "ready":
            overall = "degraded" if stats["stuck_workers"] else "ok"
        else:
            overall = state["status"]  # loading | unavailable
        return {
            "status": overall,
            "model": {"status": state["status"], "error": state["error"]},
            "gpu": gpu,
            "queue": stats,
            "limits": {
                "max_job_timeout": settings.max_job_timeout,
                "upload_max_size": settings.upload_max_size,
                "result_ttl": settings.result_ttl,
            },
        }

    async def _read_limited(upload: UploadFile) -> bytes:
        data = await upload.read(settings.upload_max_size + 1)
        await upload.close()
        if len(data) > settings.upload_max_size:
            raise InvalidImage("Ảnh vượt quá dung lượng cho phép")
        return data

    def _decode_bounded(data: bytes):
        img = decode_rgb(data, settings.max_image_pixels)
        # Chỉ giữ bản thu nhỏ vừa đủ (<= 2x kích thước inference) trong hàng đợi để tiết kiệm RAM.
        img.thumbnail((settings.width * 2, settings.height * 2))
        return img

    @app.post("/try-on")
    async def try_on(
        person_image: UploadFile = File(...),
        cloth_image: UploadFile = File(...),
        cloth_type: str = Form("upper"),
        job_id: Optional[str] = Form(None),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
    ):
        if cloth_type not in CLOTH_TYPES:
            return _err(400, "INVALID_CLOTH_TYPE", "cloth_type phải là upper, lower hoặc overall")
        if job_id is not None and not JOB_ID_RE.match(job_id):
            return _err(400, "INVALID_JOB_ID", "job_id không hợp lệ")
        if job_id is not None:
            existing = queue.get(job_id)
            if existing is not None:
                await person_image.close()
                await cloth_image.close()
                return JSONResponse(existing, status_code=200)
        steps = settings.default_steps if num_inference_steps is None else num_inference_steps
        if not settings.min_steps <= steps <= settings.max_steps:
            return _err(400, "INVALID_PARAMS", f"num_inference_steps phải trong [{settings.min_steps}, {settings.max_steps}]")
        guidance = settings.default_guidance if guidance_scale is None else guidance_scale
        if not 1.0 <= guidance <= 7.5:
            return _err(400, "INVALID_PARAMS", "guidance_scale phải trong [1.0, 7.5]")
        seed_value = settings.default_seed if seed is None else seed
        if not -1 <= seed_value < 2**31:
            return _err(400, "INVALID_PARAMS", "seed không hợp lệ")
        if state["engine"] is None:
            return _err(503, "MODEL_NOT_READY", "Model chưa sẵn sàng", {"Retry-After": "30"})

        try:
            person_bytes = await _read_limited(person_image)
            cloth_bytes = await _read_limited(cloth_image)
            person = await run_in_threadpool(_decode_bounded, person_bytes)
            del person_bytes
            cloth = await run_in_threadpool(_decode_bounded, cloth_bytes)
            del cloth_bytes
        except InvalidImage as exc:
            return _err(400, "INVALID_IMAGE", str(exc))

        try:
            job, created = queue.submit(job_id, person, cloth, cloth_type, steps, guidance, seed_value)
        except QueueFull as exc:
            return _err(429, exc.code, "Hàng đợi đã đầy", {"Retry-After": "30"})
        except NotReady:
            return _err(503, "MODEL_NOT_READY", "Model chưa sẵn sàng", {"Retry-After": "30"})
        return JSONResponse(queue.get(job.id), status_code=202 if created else 200)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        info = queue.get(job_id)
        if info is None:
            return _err(404, "JOB_NOT_FOUND", "Không tìm thấy tác vụ")
        return info

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        status, data = queue.take_result(job_id)
        if status == "not_found":
            return _err(404, "JOB_NOT_FOUND", "Không tìm thấy tác vụ")
        if status == "not_ready":
            return _err(409, "RESULT_NOT_READY", "Tác vụ chưa hoàn thành")
        if status == "expired":
            return _err(410, "RESULT_EXPIRED", "Ảnh kết quả đã hết hạn hoặc đã được giải phóng")
        return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.delete("/jobs/{job_id}")
    def job_delete(job_id: str):
        outcome = queue.cancel_or_release(job_id)
        if outcome is None:
            return _err(404, "JOB_NOT_FOUND", "Không tìm thấy tác vụ")
        return {"outcome": outcome, "job": queue.get(job_id)}

    return app


def _real_engine(settings: Settings):
    from tryon_api.engine import CatVTONEngine
    return CatVTONEngine(settings)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(Path(__file__).with_name(".env"))
    app = create_app(settings, _real_engine)
    # Một tiến trình duy nhất: model chỉ được tải một lần, hàng đợi nằm trong tiến trình này.
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()

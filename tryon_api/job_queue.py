"""Hàng đợi tác vụ thử đồ cho GPU.

- Một (hoặc MAX_CONCURRENT_INFERENCE) luồng worker duy nhất chạy inference; model dùng chung.
- Hàng đợi có giới hạn, mọi thao tác kiểm tra/ghi đều nằm trong cùng một khóa nên
  nhiều request đồng thời không thể làm hàng đợi vượt giới hạn.
- Ảnh đầu vào và ảnh kết quả chỉ nằm trong RAM; ảnh đầu vào bị bỏ ngay khi job kết thúc,
  ảnh kết quả bị bỏ khi hết hạn, khi Shop_API báo đã nhận, hoặc khi bị hủy.
- Không phụ thuộc torch: phần inference nằm trong Engine được truyền vào.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from PIL import Image

log = logging.getLogger("catvton.queue")

QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED = "queued", "processing", "completed", "failed", "cancelled"
TERMINAL = {COMPLETED, FAILED, CANCELLED}
CLOTH_TYPES = {"upper", "lower", "overall"}


class JobCancelled(Exception):
    """Ném ra từ callback để dừng vòng lặp denoising (hủy)."""


class JobTimedOut(Exception):
    """Ném ra từ callback khi vượt MAX_JOB_TIMEOUT."""


class EngineError(Exception):
    """Lỗi có mã từ Engine (ví dụ GPU_OOM)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class QueueFull(Exception):
    def __init__(self, code: str = "QUEUE_FULL") -> None:
        super().__init__(code)
        self.code = code


class NotReady(Exception):
    pass


class Engine(Protocol):
    def run(self, person: Image.Image, cloth: Image.Image, cloth_type: str, steps: int,
            guidance: float, seed: int, on_step: Callable[[int, int], None]) -> Image.Image: ...

    def encode(self, image: Image.Image) -> bytes: ...


@dataclass
class Job:
    id: str
    cloth_type: str
    steps: int
    guidance: float
    seed: int
    created_at: float
    person: Optional[Image.Image] = None
    cloth: Optional[Image.Image] = None
    status: str = QUEUED
    error_code: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    deadline: Optional[float] = None
    step: int = 0
    total_steps: int = 0
    result: Optional[bytes] = None
    result_expires_at: Optional[float] = None
    delivered_at: Optional[float] = None
    result_released: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobQueue:
    def __init__(self, engine_getter: Callable[[], Optional[Engine]], *, max_concurrent: int,
                 max_queue_size: int, job_timeout: int, result_ttl: int, delivered_grace: int,
                 job_retention: int, max_stored_results: int, clock: Callable[[], float] = time.monotonic,
                 reaper_interval: float = 5.0) -> None:
        self._engine_getter = engine_getter
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.job_timeout = job_timeout
        self.result_ttl = result_ttl
        self.delivered_grace = delivered_grace
        self.job_retention = job_retention
        self.max_stored_results = max_stored_results
        self._clock = clock
        self._reaper_interval = reaper_interval
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []        # các job đang queued theo thứ tự
        self._stop = False
        self._threads: list[threading.Thread] = []
        self.stuck_workers = 0

    # ------------------------------------------------------------------ vòng đời
    def start(self) -> None:
        for i in range(self.max_concurrent):
            t = threading.Thread(target=self._worker, name=f"tryon-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        t = threading.Thread(target=self._reaper, name="tryon-reaper", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            for job in self._jobs.values():
                job.cancel_event.set()
            self._cv.notify_all()

    # ------------------------------------------------------------------ API công khai
    def submit(self, job_id: Optional[str], person: Image.Image, cloth: Image.Image, cloth_type: str,
               steps: int, guidance: float, seed: int) -> tuple[Job, bool]:
        """Trả về (job, created). Nếu job_id đã tồn tại thì trả lại job cũ (idempotent)."""
        if cloth_type not in CLOTH_TYPES:
            raise ValueError("cloth_type không hợp lệ")
        if self._engine_getter() is None:
            raise NotReady()
        job_id = job_id or uuid.uuid4().hex
        with self._cv:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return existing, False
            if len(self._order) >= self.max_queue_size:
                raise QueueFull("QUEUE_FULL")
            if self._stored_undelivered() >= self.max_stored_results:
                raise QueueFull("RESULT_STORE_FULL")
            job = Job(id=job_id, cloth_type=cloth_type, steps=steps, guidance=guidance, seed=seed,
                      created_at=self._clock(), person=person, cloth=cloth)
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._cv.notify()
            return job, True

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else self._public(job)

    def take_result(self, job_id: str) -> tuple[str, Optional[bytes]]:
        """Trả về (trạng_thái, bytes). trạng_thái: ok | not_found | not_ready | expired."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found", None
            if job.status != COMPLETED:
                return "not_ready", None
            if job.result is None:
                return "expired", None
            now = self._clock()
            if job.delivered_at is None:
                job.delivered_at = now
            # Đã giao ít nhất một lần: chỉ giữ thêm một khoảng ngắn để frontend có thể tải lại/tải về.
            grace_end = now + self.delivered_grace
            if job.result_expires_at is None or grace_end < job.result_expires_at:
                job.result_expires_at = grace_end
            return "ok", job.result

    def cancel_or_release(self, job_id: str) -> Optional[str]:
        """Hủy job đang chờ, yêu cầu dừng job đang chạy, hoặc giải phóng ảnh kết quả.

        Trả về: cancelled | cancel_requested | released | already_finished | None (không tồn tại).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == QUEUED:
                self._order.remove(job_id)
                self._finish(job, CANCELLED, "CANCELLED")
                return "cancelled"
            if job.status == PROCESSING:
                job.cancel_event.set()
                return "cancel_requested"
            if job.status == COMPLETED and job.result is not None:
                job.result = None
                job.result_released = True
                return "released"
            return "already_finished"

    def stats(self) -> dict:
        with self._lock:
            processing = sum(1 for j in self._jobs.values() if j.status == PROCESSING)
            return {
                "processing": processing,
                "queued": len(self._order),
                "max_queue_size": self.max_queue_size,
                "max_concurrent_inference": self.max_concurrent,
                "results_in_memory": sum(1 for j in self._jobs.values() if j.result is not None),
                "stuck_workers": self.stuck_workers,
            }

    # ------------------------------------------------------------------ nội bộ
    def _stored_undelivered(self) -> int:
        return sum(1 for j in self._jobs.values() if j.result is not None and j.delivered_at is None)

    def _public(self, job: Job) -> dict:
        position = None
        if job.status == QUEUED:
            try:
                position = self._order.index(job.id) + 1
            except ValueError:
                position = None
        return {
            "job_id": job.id,
            "status": job.status,
            "queue_position": position,
            "error_code": job.error_code,
            "step": job.step if job.status == PROCESSING else None,
            "total_steps": job.total_steps if job.status == PROCESSING else None,
            "result_available": job.result is not None,
            "result_released": job.result_released,
        }

    def _finish(self, job: Job, status: str, error_code: Optional[str] = None,
                result: Optional[bytes] = None) -> None:
        """Phải gọi khi đang giữ khóa."""
        job.status = status
        job.error_code = error_code
        job.finished_at = self._clock()
        job.person = None   # bỏ ảnh đầu vào khỏi RAM ngay
        job.cloth = None
        if result is not None:
            job.result = result
            job.result_expires_at = job.finished_at + self.result_ttl
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        stored = [j for j in self._jobs.values() if j.result is not None]
        if len(stored) <= self.max_stored_results:
            return
        # Ưu tiên bỏ ảnh đã giao, cũ nhất trước.
        stored.sort(key=lambda j: (j.delivered_at is None, j.finished_at or 0))
        for j in stored[: len(stored) - self.max_stored_results]:
            j.result = None

    def _next_job(self) -> Optional[Job]:
        with self._cv:
            while not self._stop and not self._order:
                self._cv.wait(timeout=1.0)
            if self._stop:
                return None
            job = self._jobs[self._order.pop(0)]
            job.status = PROCESSING
            job.started_at = self._clock()
            job.deadline = job.started_at + self.job_timeout
            job.total_steps = job.steps
            return job

    def _worker(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        engine = self._engine_getter()

        def check(step: int = 0, total: int = 0) -> None:
            if step:
                job.step, job.total_steps = step, total
            if job.cancel_event.is_set():
                raise JobCancelled()
            if job.deadline is not None and self._clock() > job.deadline:
                raise JobTimedOut()

        status, code, png = FAILED, "INFERENCE_ERROR", None
        try:
            if engine is None:
                raise EngineError("MODEL_NOT_READY")
            check()
            image = engine.run(job.person, job.cloth, job.cloth_type, job.steps, job.guidance,
                               job.seed, check)
            check()
            png = engine.encode(image)
            del image
            status, code = COMPLETED, None
        except JobCancelled:
            status, code = CANCELLED, "CANCELLED"
        except JobTimedOut:
            status, code = FAILED, "TIMEOUT"
        except EngineError as exc:
            status, code = FAILED, exc.code
            log.warning("Job %s lỗi engine: %s", job.id, exc.code)
        except Exception as exc:  # không để một job lỗi làm dừng worker
            status, code = FAILED, "INFERENCE_ERROR"
            log.error("Job %s lỗi inference: %s", job.id, type(exc).__name__)
            log.debug("Chi tiết", exc_info=True)
        with self._lock:
            if job.status == PROCESSING:
                self._finish(job, status, code, png)
            else:
                # Watchdog đã đánh dấu job hết hạn trong lúc worker còn chạy -> bỏ kết quả.
                self.stuck_workers = max(0, self.stuck_workers - 1)
                job.person = job.cloth = None

    def _reaper(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                self._cv.wait(timeout=self._reaper_interval)
                if self._stop:
                    return
            self.reap()

    def reap(self) -> None:
        """Dọn ảnh hết hạn, metadata cũ và phát hiện worker bị treo. Có thể gọi trực tiếp (test)."""
        now = self._clock()
        with self._lock:
            for job_id in list(self._jobs):
                job = self._jobs[job_id]
                if job.status == PROCESSING and job.deadline is not None and now > job.deadline + 30:
                    # Inference không phản hồi callback (ví dụ kẹt trong một lệnh CUDA dài).
                    log.error("Job %s vượt quá thời gian, worker có thể bị treo", job_id)
                    self.stuck_workers += 1
                    self._finish(job, FAILED, "TIMEOUT")
                    job.cancel_event.set()
                if job.result is not None and job.result_expires_at is not None and now >= job.result_expires_at:
                    job.result = None
                if job.status in TERMINAL and job.finished_at is not None and now - job.finished_at > self.job_retention:
                    del self._jobs[job_id]

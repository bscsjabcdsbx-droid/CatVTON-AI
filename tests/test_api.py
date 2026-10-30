"""Kiểm thử hàng đợi / API mà không cần GPU.

FakeEngine CHỈ dùng trong test để kiểm tra logic hàng đợi, giới hạn, timeout, hủy, xác thực.
Nó không bao giờ được nạp bởi api_server.main(); production luôn dùng CatVTONEngine thật.
Chạy:  python -m pytest tests -q
"""
from __future__ import annotations

import io
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_server import create_app  # noqa: E402
from tryon_api.job_queue import EngineError, JobQueue  # noqa: E402
from tryon_api.settings import load_settings  # noqa: E402

KEY = "test-key-0123456789abcdefghij"
H = {"X-API-Key": KEY}


def png_bytes(w=300, h=400, color=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeEngine:  # chỉ dùng cho test
    def __init__(self, step_delay=0.0, fail_with=None, gate: threading.Event | None = None, hang=False):
        self.step_delay = step_delay
        self.fail_with = fail_with
        self.gate = gate
        self.hang = hang
        self.calls = 0

    def run(self, person, cloth, cloth_type, steps, guidance, seed, on_step):
        self.calls += 1
        if self.fail_with is not None and self.calls == 1:
            raise self.fail_with
        if self.hang:
            time.sleep(3)
            return person
        for i in range(steps):
            if self.gate is not None:
                self.gate.wait(5)
            if self.step_delay:
                time.sleep(self.step_delay)
            on_step(i + 1, steps)
        return Image.new("RGB", (64, 64), (1, 2, 3))

    def encode(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("CATVTON_API_KEY", KEY)
    return replace(load_settings(), min_steps=1, default_steps=3, max_job_timeout=10)


def make_client(settings, engine):
    app = create_app(settings, lambda s: engine)
    client = TestClient(app)
    client.__enter__()
    deadline = time.time() + 5
    while app.state.model_state["status"] != "ready" and time.time() < deadline:
        time.sleep(0.02)
    return app, client


def submit(client, job_id=None, person=None, cloth_type="upper", **extra):
    files = {"person_image": ("p.png", person or png_bytes(), "image/png"),
             "cloth_image": ("c.png", png_bytes(color=(0, 0, 200)), "image/png")}
    data = {"cloth_type": cloth_type, **extra}
    if job_id:
        data["job_id"] = job_id
    return client.post("/try-on", headers=H, files=files, data=data)


def wait_status(client, job_id, statuses, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}", headers=H).json()
        if r["status"] in statuses:
            return r
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} không đạt {statuses}: {r}")


def test_requires_api_key(settings):
    _, c = make_client(settings, FakeEngine())
    assert c.get("/livez").status_code == 200
    assert c.get("/health").status_code == 401
    assert c.get("/health", headers={"X-API-Key": "sai"}).status_code == 401
    r = c.post("/try-on", files={"person_image": ("p.png", png_bytes(), "image/png")})
    assert r.status_code == 401


def test_refuses_to_start_without_key(settings):
    with pytest.raises(RuntimeError):
        create_app(replace(settings, api_key=""), lambda s: FakeEngine())


def test_full_flow_result_bytes_and_release(settings):
    _, c = make_client(settings, FakeEngine())
    r = submit(c, job_id="job-aaaaaaaa")
    assert r.status_code == 202, r.text
    wait_status(c, "job-aaaaaaaa", {"completed"})
    res = c.get("/jobs/job-aaaaaaaa/result", headers=H)
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"
    # vẫn lấy lại được trong thời gian ân hạn (để frontend tải về)
    assert c.get("/jobs/job-aaaaaaaa/result", headers=H).status_code == 200
    d = c.delete("/jobs/job-aaaaaaaa", headers=H).json()
    assert d["outcome"] == "released"
    assert c.get("/jobs/job-aaaaaaaa/result", headers=H).status_code == 410
    assert c.get("/health", headers=H).json()["queue"]["results_in_memory"] == 0


def test_idempotent_job_id(settings):
    _, c = make_client(settings, FakeEngine())
    assert submit(c, job_id="same-job-id1").status_code == 202
    r2 = submit(c, job_id="same-job-id1")
    assert r2.status_code == 200 and r2.json()["job_id"] == "same-job-id1"


def test_queue_limit_under_concurrency(settings):
    gate = threading.Event()  # giữ job đầu tiên ở trạng thái processing
    app, c = make_client(replace(settings, max_queue_size=5), FakeEngine(gate=gate))
    codes = []
    lock = threading.Lock()

    def one(i):
        r = submit(c, job_id=f"conc-job-{i:04d}")
        with lock:
            codes.append(r.status_code)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    accepted = codes.count(202)
    stats = c.get("/health", headers=H).json()["queue"]
    assert stats["queued"] <= 5
    assert stats["processing"] <= 1
    assert accepted <= 6, codes             # 1 đang chạy + tối đa 5 chờ
    assert codes.count(429) == 20 - accepted
    gate.set()


def test_queue_position_and_cancel_queued(settings):
    gate = threading.Event()
    _, c = make_client(settings, FakeEngine(gate=gate))
    submit(c, job_id="first-job-01")
    wait_status(c, "first-job-01", {"processing"})
    submit(c, job_id="second-job-1")
    submit(c, job_id="third-job-01")
    assert c.get("/jobs/third-job-01", headers=H).json()["queue_position"] == 2
    assert c.delete("/jobs/second-job-1", headers=H).json()["outcome"] == "cancelled"
    assert c.get("/jobs/third-job-01", headers=H).json()["queue_position"] == 1
    gate.set()
    wait_status(c, "third-job-01", {"completed"})


def test_cancel_processing(settings):
    _, c = make_client(replace(settings, default_steps=50, max_steps=50), FakeEngine(step_delay=0.02))
    submit(c, job_id="cancel-me-01")
    wait_status(c, "cancel-me-01", {"processing"})
    assert c.delete("/jobs/cancel-me-01", headers=H).json()["outcome"] == "cancel_requested"
    r = wait_status(c, "cancel-me-01", {"cancelled"})
    assert r["error_code"] == "CANCELLED"


def test_timeout_does_not_block_queue(settings):
    app, c = make_client(replace(settings, default_steps=50, max_steps=50), FakeEngine(step_delay=0.05))
    app.state.queue.job_timeout = 0.3
    submit(c, job_id="slow-job-001")
    r = wait_status(c, "slow-job-001", {"failed"}, timeout=10)
    assert r["error_code"] == "TIMEOUT"
    app.state.queue.job_timeout = 30
    submit(c, job_id="after-timeo1", num_inference_steps="2")
    wait_status(c, "after-timeo1", {"completed"})


def test_oom_is_reported_and_worker_survives(settings):
    _, c = make_client(settings, FakeEngine(fail_with=EngineError("GPU_OOM")))
    submit(c, job_id="oom-job-0001")
    assert wait_status(c, "oom-job-0001", {"failed"})["error_code"] == "GPU_OOM"
    submit(c, job_id="next-job-001")
    wait_status(c, "next-job-001", {"completed"})


def test_unexpected_exception_does_not_kill_worker(settings):
    _, c = make_client(settings, FakeEngine(fail_with=KeyError("boom")))
    submit(c, job_id="crash-job-01")
    assert wait_status(c, "crash-job-01", {"failed"})["error_code"] == "INFERENCE_ERROR"
    submit(c, job_id="next-job-002")
    wait_status(c, "next-job-002", {"completed"})


def test_invalid_images(settings):
    _, c = make_client(settings, FakeEngine())
    r = submit(c, person=b"this is not an image at all")
    assert r.status_code == 400 and r.json()["error_code"] == "INVALID_IMAGE"
    # file PNG giả mạo (đúng đuôi, sai nội dung)
    r = submit(c, person=b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    assert r.status_code == 400
    big = replace(settings, upload_max_size=2000)
    _, c2 = make_client(big, FakeEngine())
    r = submit(c2, person=png_bytes(1000, 1000, (10, 20, 30)) + b"x" * 5000)
    assert r.status_code in (400, 413)
    assert submit(c, cloth_type="hat").status_code == 400
    assert submit(c, num_inference_steps="999").status_code == 400


def test_model_not_ready(settings):
    def failing_factory(s):
        raise RuntimeError("CUDA_UNAVAILABLE")
    app = create_app(settings, failing_factory)
    with TestClient(app) as c:
        time.sleep(0.2)
        h = c.get("/health", headers=H).json()
        assert h["status"] == "unavailable" and h["model"]["error"] == "CUDA_UNAVAILABLE"
        r = submit(c)
        assert r.status_code == 503 and r.json()["error_code"] == "MODEL_NOT_READY"


def test_result_expires_and_stuck_worker_reaped():
    now = [1000.0]
    eng = FakeEngine()
    q = JobQueue(lambda: eng, max_concurrent=1, max_queue_size=5, job_timeout=10, result_ttl=60,
                 delivered_grace=5, job_retention=100, max_stored_results=3, clock=lambda: now[0])
    img = Image.new("RGB", (64, 64))
    job, _ = q.submit("exp-job-0001", img, img, "upper", 2, 2.5, 1)
    q._run_one(q._next_job())
    assert q.get("exp-job-0001")["result_available"]
    assert q.get("exp-job-0001")["status"] == "completed"
    assert job.person is None and job.cloth is None   # ảnh đầu vào đã bị bỏ khỏi RAM
    now[0] += 61
    q.reap()
    assert q.take_result("exp-job-0001")[0] == "expired"
    now[0] += 200
    q.reap()
    assert q.get("exp-job-0001") is None                # metadata cũng bị dọn

    # Worker treo: watchdog đánh dấu TIMEOUT để job không kẹt mãi
    q.submit("stuck-job-01", img, img, "upper", 2, 2.5, 1)
    j = q._next_job()
    now[0] += 45
    q.reap()
    assert q.get("stuck-job-01")["status"] == "failed"
    assert q.get("stuck-job-01")["error_code"] == "TIMEOUT"
    assert q.stats()["stuck_workers"] == 1
    q._run_one(j)    # worker cuối cùng cũng trả về -> kết quả bị bỏ
    assert q.stats()["stuck_workers"] == 0
    assert not q.get("stuck-job-01")["result_available"]

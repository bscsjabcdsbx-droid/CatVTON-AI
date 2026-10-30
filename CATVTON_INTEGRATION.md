# Tích hợp CatVTON (thử đồ AI) vào Shop_API

Tài liệu này áp dụng cho hai repository đã chỉnh sửa: **Shop_API** (website + backend) và **CatVTON-AI** (máy AI).
Bản sao giống hệt nhau nằm ở gốc của cả hai repository.

## 0. Kiến trúc

```
Trình duyệt ──(JWT)──▶ Cloudflare Pages (public/)            shop-9be.pages.dev
     │
     └──(JWT, multipart)──▶ Render: Shop_API (Express)        /api/try-on/...
                               │  PostgreSQL: chỉ metadata TryOnJob (KHÔNG có ảnh)
                               └──(HTTPS + X-API-Key)──▶ Cloudflare Tunnel ──▶ 127.0.0.1:8000
                                                          Máy Windows: CatVTON API (FastAPI)
                                                          1 worker GPU + hàng đợi trong RAM
```

- Frontend **không bao giờ** gọi CatVTON hay Tunnel. Frontend chỉ gửi `productId` + ảnh người.
- Backend tự lấy ảnh sản phẩm từ database, kiểm tra quyền, giới hạn lượt, rồi gửi sang CatVTON.
- CatVTON trả `202` ngay, xử lý lần lượt trên GPU. Backend hỏi trạng thái mỗi khi frontend kiểm tra (khoảng 2 giây/lần).
  Khi xong, `GET /api/try-on/:id/result` **stream thẳng bytes PNG** từ RAM của CatVTON tới trình duyệt.
- Ảnh người dùng: nằm trong RAM của Render trong lúc chuyển tiếp, rồi trong RAM của CatVTON tới khi job kết thúc.
  Ảnh kết quả: nằm trong RAM của CatVTON, bị xóa khi (a) người dùng đóng cửa sổ sau khi xem, (b) 5 phút sau lần lấy đầu tiên,
  hoặc (c) hết `TRYON_RESULT_TTL` (mặc định 60 phút) nếu không ai lấy. Không có ảnh nào được ghi vào PostgreSQL hay ổ đĩa của Render.

### Trạng thái job

| Trạng thái | Ý nghĩa |
|---|---|
| `queued` | Đang chờ GPU (có vị trí thật trong hàng đợi) |
| `processing` | GPU đang chạy |
| `completed` | Có ảnh kết quả (trong thời gian lưu tạm) |
| `failed` | Thất bại; `errorCode` cho biết lý do (TIMEOUT, GPU_OOM, SERVICE_OFFLINE, JOB_LOST, ...) |
| `cancelled` | Người dùng đã hủy |

### Endpoint

**Shop_API** (frontend gọi):

| Method | Đường dẫn | Quyền |
|---|---|---|
| GET | `/api/try-on/config` | công khai: bật/tắt, máy AI online, dung lượng tối đa |
| GET | `/api/try-on/active` | đăng nhập: job đang chạy (để khôi phục khi tải lại trang) + hạn mức còn lại |
| POST | `/api/try-on` | đăng nhập: `multipart/form-data` gồm `productId`, `personImage` + header `Idempotency-Key` |
| GET | `/api/try-on/:jobId/status` | chủ job |
| GET | `/api/try-on/:jobId/result` | chủ job: **trả ảnh PNG trực tiếp** |
| DELETE | `/api/try-on/:jobId` | chủ job: hủy job đang chờ / giải phóng ảnh kết quả |

**CatVTON API** (chỉ backend gọi, bắt buộc header `X-API-Key`):
`GET /livez` (không cần key), `GET /health`, `POST /try-on`, `GET /jobs/{id}`, `GET /jobs/{id}/result`, `DELETE /jobs/{id}`.

> **Thay đổi không tương thích ngược:** `POST /try-on` của `api_server.py` cũ trả ảnh ngay (đồng bộ). Bản mới trả `202` + trạng thái job.
> `src/routes/tryOn.ts` cũ (chưa từng được gắn vào `index.ts`) đã được viết lại cho khớp. Giao diện Gradio (`app.py`) không bị thay đổi.

---

## 1. Cài Python (máy Windows chạy AI)

1. README gốc của CatVTON dùng **Python 3.9**. Cài Python 3.9 hoặc 3.10 bản 64-bit từ python.org. Lúc cài, tick "Add python.exe to PATH".
2. Kiểm tra: `py -3.9 --version` (hoặc `python --version`).
3. Cài Git for Windows nếu chưa có.

## 2. Cài PyTorch có CUDA

`requirements.txt` ghim `torch==2.4.0`, `torchvision==0.19.0`. Cài **bản CUDA** trước, nếu không pip sẽ cài bản chỉ chạy CPU:

```bat
cd CatVTON-AI
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Cần cài driver NVIDIA mới (Game Ready hoặc Studio). Không cần cài CUDA Toolkit riêng vì wheel PyTorch đã kèm runtime.

## 3. Kiểm tra GPU

```bat
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kết quả phải có `True` và tên `NVIDIA GeForce RTX 4060 Laptop GPU`.
Laptop có hai GPU: trong *Windows Settings → Display → Graphics*, đặt `python.exe` của `.venv` dùng *High performance*.

## 4. Cài CatVTON

```bat
pip install -r requirements.txt -r requirements-api.txt
```

Các lỗi có thể gặp khi cài trên Windows (tôi **chưa** kiểm chứng được trên Windows):

- `setuptools==51.0.0` quá cũ. Nếu nó làm hỏng quá trình cài, xóa dòng đó khỏi `requirements.txt` rồi cài lại.
- `pycocotools` cần Visual C++ Build Tools nếu không có wheel sẵn cho phiên bản Python của bạn.

## 5. Model và file phụ thuộc

Lần chạy đầu, chương trình tự tải từ Hugging Face về `%USERPROFILE%\.cache\huggingface` (khoảng vài GB, cần mạng ổn định):

- `zhengchong/CatVTON`: trọng số attention `mix-48k-1024`, kèm checkpoint DensePose và SCHP.
- `booksforcharlie/stable-diffusion-inpainting`: bản mirror của `runwayml/stable-diffusion-inpainting`, gồm UNet, scheduler và safety checker.
- `stabilityai/sd-vae-ft-mse`: VAE, được `model/pipeline.py` tải.

Muốn dùng model đã tải sẵn ở thư mục local, đặt `CATVTON_BASE_MODEL` hoặc `CATVTON_ATTN_REPO` là đường dẫn thư mục đó.

## 6. Chạy CatVTON API ở máy local

```bat
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Dán chuỗi vừa in ra vào `CATVTON_API_KEY=` trong `.env`. Đây là khóa bí mật, **không commit**; `.env` đã có trong `.gitignore`. Sau đó:

```bat
start_api.bat
```

Hoặc chạy: `.venv\Scripts\python.exe api_server.py`.

- API chỉ lắng nghe ở `127.0.0.1:8000`, không mở port ra mạng.
- Model được tải **một lần** trong luồng nền. Trong lúc tải, `/health` báo `"loading"` và `POST /try-on` trả `503 MODEL_NOT_READY`.
- Chỉ chạy **một** tiến trình. Không dùng `--workers` vì mỗi tiến trình sẽ tải thêm một bản model vào VRAM.

**Tối ưu cho 8GB VRAM.** README gốc ghi bf16 ở 1024×768 cần khoảng 8GB, sát giới hạn của RTX 4060 Laptop. Nếu gặp `GPU_OOM`, thử lần lượt các cách sau, mỗi lần một thay đổi:

1. Đóng ứng dụng khác đang dùng GPU (game, trình duyệt có tăng tốc phần cứng, OBS).
2. `TRYON_VAE_TILING=1`
3. `TRYON_MASKER_DEVICE=cpu`: DensePose và SCHP chạy trên CPU, chậm hơn nhưng bớt VRAM.
4. `TRYON_WIDTH=576`, `TRYON_HEIGHT=768`: chất lượng giảm.
5. `TRYON_DEFAULT_STEPS=20`: nhanh hơn, nhưng không giảm VRAM đỉnh.

Sau mỗi job, kể cả khi lỗi, service gọi `gc.collect()` và `torch.cuda.empty_cache()`.
Ảnh đầu vào được thu nhỏ còn tối đa 2 lần kích thước inference ngay khi nhận, và bị bỏ khỏi RAM khi job kết thúc.

## 7. Kiểm tra endpoint health

Mở cửa sổ CMD khác:

```bat
curl http://127.0.0.1:8000/livez
curl -H "X-API-Key: <khóa trong .env>" http://127.0.0.1:8000/health
```

Đợi đến khi `"status": "ok"` và `"model": {"status": "ready"}`.

Nếu thấy `unavailable` kèm `CUDA_UNAVAILABLE`, PyTorch không thấy GPU: xem lại mục 2–3. Nếu thấy `MODEL_LOAD_FAILED`, xem log trong cửa sổ đang chạy API (thường do lỗi mạng khi tải model).

Thử tạo ảnh trực tiếp bằng ảnh mẫu trong repo:

```bat
curl -H "X-API-Key: <khóa>" -F job_id=test-job-0001 -F cloth_type=upper ^
  -F person_image=@resource/demo/example/person/men/model_5.png ^
  -F cloth_image=@resource/demo/example/condition/upper/24083449_54173465_2048.jpg ^
  http://127.0.0.1:8000/try-on
curl -H "X-API-Key: <khóa>" http://127.0.0.1:8000/jobs/test-job-0001
curl -H "X-API-Key: <khóa>" -o ketqua.png http://127.0.0.1:8000/jobs/test-job-0001/result
```

## 8. Cài Cloudflare Tunnel (cloudflared)

```bat
winget install --id Cloudflare.cloudflared
cloudflared --version
```

## 9. Tạo Tunnel

### Giai đoạn thử nghiệm: Quick Tunnel (không cần domain hay tài khoản)

```bat
start_tunnel.bat
```

Lệnh tương đương: `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000`.

Cửa sổ sẽ in ra một URL dạng `https://<tên-ngẫu-nhiên>.trycloudflare.com`. Đó là giá trị cho `CATVTON_API_URL` trên Render.

Giới hạn của Quick Tunnel:

- **URL đổi mỗi lần chạy lại.** Mỗi lần khởi động lại tunnel phải cập nhật `CATVTON_API_URL` trên Render; Render sẽ tự deploy lại.
- Cloudflare mô tả đây là tính năng dành cho thử nghiệm, không có cam kết uptime.
- Nếu `%USERPROFILE%\.cloudflared\config.yml` tồn tại, quick tunnel có thể không chạy. Đổi tên file đó trong lúc thử.
- URL này công khai. Lớp bảo vệ là `X-API-Key`: mọi request thiếu hoặc sai key đều bị trả `401` trước khi đọc ảnh.

### Sau này: Named Tunnel + domain riêng

Khi đã có domain quản lý trên Cloudflare, hostname sẽ cố định.

```bat
cloudflared tunnel login
cloudflared tunnel create catvton
cloudflared tunnel route dns catvton catvton-api.<domain-của-bạn>
```

Tạo `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: catvton
credentials-file: C:\Users\<bạn>\.cloudflared\<tunnel-id>.json
ingress:
  - hostname: catvton-api.<domain-của-bạn>
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Chạy bằng `cloudflared tunnel run catvton`. Để tự khởi động cùng Windows, dùng `cloudflared service install`.

## 10. Cloudflare Access (tùy chọn, dùng cùng Named Tunnel)

Hiện chưa cần. Khi chuyển sang Named Tunnel:

1. Vào Zero Trust → Access → Service Auth → **Service Tokens** → Create. Lưu lại *Client ID* và *Client Secret*.
2. Vào Access → Applications → Add → Self-hosted, hostname `catvton-api.<domain>`. Tạo Policy với Action **Service Auth**, Include → Service Token → chọn token vừa tạo.
3. Trên Render, đặt `CATVTON_ACCESS_CLIENT_ID` và `CATVTON_ACCESS_CLIENT_SECRET`. Backend sẽ tự gửi header `CF-Access-Client-Id` và `CF-Access-Client-Secret`.
4. Vẫn giữ `CATVTON_API_KEY`: đây là lớp xác thực thứ hai, do chính CatVTON kiểm tra.

## 11. Biến môi trường trên Render (Shop_API)

Vào Render → service → Environment. Xem đầy đủ trong `.env.example`. Tối thiểu cần:

| Biến | Giá trị |
|---|---|
| `CLIENT_ORIGIN` | `https://shop-9be.pages.dev,https://*.shop-9be.pages.dev` |
| `CATVTON_API_URL` | URL tunnel, ví dụ `https://xxxx.trycloudflare.com` (không có `/` ở cuối) |
| `CATVTON_API_KEY` | **giống hệt** khóa trong `.env` của CatVTON |
| `TRYON_GARMENT_ALLOWED_HOSTS` | Chỉ cần nếu ảnh sản phẩm là URL ngoài, ví dụ `res.cloudinary.com` |

Sau khi deploy code mới, chạy **một lần**:

```
npx prisma db push
```

Có thể chạy trong Render Shell, hoặc thêm vào Build Command. Lệnh này **chỉ thêm** bảng `TryOnJob`, enum `TryOnStatus` và cột `Product.tryOnType` (có thể null). Dữ liệu cũ không bị đụng tới.

> Nếu `prisma db push` cảnh báo có thể mất dữ liệu ("data loss"), **dừng lại, không** thêm `--accept-data-loss`, và kiểm tra lại schema.

Cloudflare Pages tự deploy lại thư mục `public/`, trong đó có file mới `tryon.js`. Không có biến môi trường nào cho frontend; frontend không chứa URL máy AI hay khóa nào.

## 12. Biến môi trường local

- **CatVTON-AI/.env**: xem `CatVTON-AI/.env.example`. Bắt buộc có `CATVTON_API_KEY`.
- **Shop_API/.env** (khi chạy backend ở máy): xem `Shop_API/.env.example`. Nếu backend và CatVTON chạy cùng một máy, đặt `CATVTON_API_URL=http://127.0.0.1:8000`.

## 13. Chạy thử trên website

1. Máy Windows: chạy `start_api.bat`, đợi `/health` báo `ok`, rồi chạy `start_tunnel.bat` và cập nhật `CATVTON_API_URL` trên Render.
2. Mở `https://<backend>.onrender.com/api/try-on/config`, phải thấy `"enabled":true,"online":true,"ready":true`.
3. Đăng nhập website, mở một sản phẩm thuộc danh mục Áo, Áo khoác, Quần hoặc Váy, bấm **Thử đồ với AI**, chọn ảnh toàn thân, xem trước, rồi bấm **Bắt đầu thử đồ**.
4. Giao diện hiện "Đang chờ đến lượt — vị trí thứ N" hoặc "AI đang xử lý ảnh…". Khi xong, ảnh kết quả hiện ngay, kèm nút **Tải ảnh về**.

**Ảnh sản phẩm:** CatVTON dùng **ảnh đầu tiên** của sản phẩm. Ảnh này phải chụp rõ đúng món đồ, tốt nhất là ảnh phẳng (flat-lay) hoặc người mẫu trên nền trơn.
Dữ liệu seed hiện dùng ảnh ngẫu nhiên từ `picsum.photos`, nên:

- Kết quả thử đồ với các ảnh đó sẽ vô nghĩa.
- Các URL đó bị chặn vì `picsum.photos` không nằm trong `TRYON_GARMENT_ALLOWED_HOSTS`.

Hãy tải ảnh sản phẩm thật lên bằng trang admin; ảnh tải lên được lưu ở `/api/uploads/:id` và dùng được ngay.

**Loại trang phục:** mặc định tự suy theo danh mục qua `TRYON_CATEGORY_MAP` (`ao` và `khoac` → upper, `quan` → lower, `vay` → overall).
Riêng từng sản phẩm có thể ghi đè trong form sửa sản phẩm ở trang admin, mục "Thử đồ AI". Ví dụ "Chân váy chữ A" nên đặt là *Quần, chân váy / phần dưới*; phụ kiện đặt *Không hỗ trợ*.

## 14. Kiểm tra hàng đợi và giới hạn

| Giới hạn | Nơi kiểm tra | Mặc định | Khi vượt |
|---|---|---|---|
| Job đang chờ/chạy mỗi tài khoản | Shop_API (khóa `pg_advisory_xact_lock`) | 1 | 429 `ACTIVE_JOB_EXISTS` |
| Lượt trong cửa sổ thời gian | Shop_API | 3 lượt / 10 phút | 429 `RATE_LIMITED` + `Retry-After` |
| Lượt trong 24 giờ | Shop_API | 10 | 429 `DAILY_LIMIT` + `Retry-After` |
| Request API mỗi phút (kể cả kiểm tra trạng thái) | Shop_API | 60 | 429 `TOO_MANY_REQUESTS` |
| Hàng đợi GPU | CatVTON (một khóa chung) | 1 chạy + 5 chờ | 503 `QUEUE_FULL` |
| Thời gian một job trên GPU | CatVTON (kiểm tra sau mỗi bước denoising) | 180 giây | `failed` / `TIMEOUT` |

- Lượt **được tính**: job đã gửi thành công (đang chờ, đang chạy, hoàn thành, bị người dùng hủy) và job lỗi `INFERENCE_ERROR`.
- Lượt **không bị tính**: lỗi do hệ thống (máy AI tắt, hàng đợi đầy, TIMEOUT, GPU_OOM, JOB_LOST, model chưa sẵn sàng) và lỗi đầu vào (file hỏng, sai định dạng, sản phẩm không hỗ trợ).
- Gửi lại cùng một `Idempotency-Key` (lỗi mạng, double-click) sẽ trả lại job cũ, không tạo job mới.

Xem hàng đợi trên máy AI: `curl -H "X-API-Key: <khóa>" http://127.0.0.1:8000/health`, phần `queue`.

## 15. Lỗi GPU, CUDA, timeout, Tunnel

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| Website báo "Máy chủ AI hiện không hoạt động" | Máy tắt, mất mạng, tunnel tắt, hoặc URL quick tunnel đã đổi. Mở lại API và tunnel, cập nhật `CATVTON_API_URL`. Các chức năng mua sắm khác vẫn chạy bình thường. |
| `/api/try-on/config` báo `online:false` | Như trên. Thử `curl <CATVTON_API_URL>/livez` từ máy khác. |
| Log Render có "CatVTON từ chối API key" | `CATVTON_API_KEY` ở hai phía không giống nhau. |
| `failed` / `GPU_OOM` | Xem mục 6, phần tối ưu VRAM. Service vẫn chạy tiếp, job sau không bị ảnh hưởng. |
| `failed` / `TIMEOUT` | Job vượt `MAX_JOB_TIMEOUT`. GPU có thể đang chạy ở chế độ tiết kiệm điện: cắm sạc laptop, chọn chế độ High performance. |
| `/health` báo `degraded` (`stuck_workers` > 0) | Một lệnh CUDA bị treo, không trả về. Khởi động lại CatVTON API (mục 16). |
| `failed` / `JOB_LOST` | CatVTON đã khởi động lại nên hàng đợi trong RAM bị mất. Người dùng thử lại; lượt này không bị tính. |
| Job kẹt `queued`/`processing` | Không xảy ra lâu: quá `TRYON_JOB_MAX_AGE` (20 phút), backend tự đánh dấu `TIMEOUT`. |
| Render khởi động lại khi đang có job | Trạng thái nằm trong PostgreSQL, job trên máy AI vẫn chạy. Khi người dùng mở lại, giao diện tự khôi phục. |

## 16. Khởi động lại service

- **CatVTON:** đóng cửa sổ `start_api.bat` bằng Ctrl+C, rồi chạy lại. Job đang chờ sẽ mất và được báo `JOB_LOST`.
- **Quick Tunnel:** chạy lại `start_tunnel.bat` → **cập nhật `CATVTON_API_URL` trên Render**.
- **Shop_API:** Render → Manual Deploy → Restart. Dữ liệu job vẫn còn trong database.

## 17. Dừng service

- Ctrl+C ở cửa sổ CatVTON API và cửa sổ tunnel. Website vẫn chạy; nút thử đồ sẽ báo máy AI không hoạt động.
- Muốn tắt hẳn tính năng: đặt `TRYON_ENABLED=false` trên Render. Nút sẽ bị ẩn và API trả 503.

## 18. Điều chỉnh giới hạn lượt

Chỉ cần sửa biến môi trường, không cần sửa code:

- **Trên Render:** `TRYON_USER_MAX_ACTIVE_JOBS`, `TRYON_RATE_LIMIT_WINDOW` (phút), `TRYON_RATE_LIMIT_MAX`, `TRYON_DAILY_LIMIT`, `TRYON_REQUESTS_PER_MINUTE`.
- **Trên máy AI (`.env`):** `TRYON_MAX_QUEUE_SIZE`, `MAX_JOB_TIMEOUT`, `TRYON_RESULT_TTL`.

Với 8GB VRAM, **giữ `MAX_CONCURRENT_INFERENCE=1`**. Các worker dùng chung một model và chạy song song trên cùng GPU, nên tăng lên gần như chắc chắn gây hết VRAM.

## 19. Kiểm tra và dọn tác vụ cũ

- Backend tự dọn mỗi `TRYON_CLEANUP_INTERVAL` giây. Việc dọn gồm hỏi lại trạng thái các job chưa xong, đánh dấu job quá hạn là `TIMEOUT`, và xóa metadata cũ hơn `TRYON_JOB_RETENTION_HOURS` (48 giờ).
- CatVTON tự dọn mỗi 5 giây: xóa ảnh kết quả hết hạn, và xóa thông tin job đã kết thúc sau `TRYON_JOB_RETENTION`.
- Xem bằng SQL:

```sql
SELECT status, "errorCode", count(*) FROM "TryOnJob" GROUP BY 1, 2;
SELECT * FROM "TryOnJob" WHERE status IN ('QUEUED','PROCESSING') ORDER BY "createdAt";
```

- Xóa thủ công metadata (không có ảnh nào trong DB để xóa): `DELETE FROM "TryOnJob" WHERE "createdAt" < now() - interval '2 days';`

---

## 20. Quyền riêng tư và lưu trữ dữ liệu

| Dữ liệu | Ở đâu | Bao lâu |
|---|---|---|
| Ảnh người dùng tải lên | RAM của Render khi đang chuyển tiếp; RAM của CatVTON | Tới khi job kết thúc (xong, lỗi hoặc hủy) |
| Ảnh kết quả | RAM của CatVTON | Xóa khi người dùng đóng cửa sổ sau khi xem, hoặc 5 phút sau lần lấy đầu, hoặc tối đa `TRYON_RESULT_TTL` (60 phút) |
| Metadata job (id, user, sản phẩm, trạng thái, thời gian) | PostgreSQL | `TRYON_JOB_RETENTION_HOURS` (48 giờ) |

- Trình duyệt vẽ lại ảnh qua canvas trước khi gửi, nên metadata EXIF (kể cả vị trí GPS) bị loại bỏ.
- `api_server.py` nâng ngưỡng đệm multipart của Starlette để ảnh upload ở lại trong RAM thay vì bị ghi ra file tạm. Nếu phiên bản Starlette mà gradio kéo về không hỗ trợ việc này, log khởi động sẽ in cảnh báo; khi đó ảnh lớn hơn 1MB có thể bị đệm tạm ra ổ đĩa trong lúc nhận.
- Không có URL công khai nào trỏ tới ảnh. Ảnh chỉ lấy được qua API, có kiểm tra chủ sở hữu.
- Ảnh không được dùng để huấn luyện, và không lưu trong thư mục public.
- Muốn xóa ngay: người dùng bấm Đóng (ảnh kết quả bị giải phóng), hoặc dừng CatVTON (toàn bộ RAM được giải phóng).

## 21. License (phải đọc trước khi dùng thật)

| Thành phần | License | Ghi chú |
|---|---|---|
| CatVTON: mã nguồn + checkpoint `zhengchong/CatVTON` | **CC BY-NC-SA 4.0** (file `LICENSE`, README) | **Cấm thương mại.** Phải ghi công tác giả và giữ cùng license cho phần sửa đổi. Đồ án học tập không bán hàng thật thì nằm trong phạm vi NonCommercial. |
| `runwayml/stable-diffusion-inpainting` (bản mirror `booksforcharlie/...`) | CreativeML OpenRAIL-M | Có danh sách mục đích sử dụng bị cấm. Repo mirror không phải nguồn chính thức. |
| `stabilityai/sd-vae-ft-mse` | Xem model card trên Hugging Face | Chưa được tôi kiểm chứng |
| DensePose / detectron2 (thư mục trong repo) | Apache-2.0 theo detectron2 gốc | Chưa được tôi kiểm chứng trên bản trong repo |
| SCHP (Self-Correction Human Parsing) | Kiểm tra repo gốc | Chưa được tôi kiểm chứng |

**Nếu dùng cho website bán hàng thật, phải xin giấy phép thương mại từ tác giả CatVTON** hoặc thay bằng model có license phù hợp.
Bộ lọc NSFW (safety checker) của pipeline gốc vẫn được bật (`TRYON_SKIP_SAFETY_CHECK=0`).
Trên website nên ghi rõ "Ảnh do AI tạo, chỉ mang tính minh họa"; giao diện đã có dòng này.

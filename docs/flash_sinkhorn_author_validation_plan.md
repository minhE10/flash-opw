# Kế hoạch kiểm tra FlashSinkhorn author

Nguồn ghim: `bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1` (0.4.1).
Reference gốc chỉ đọc; 131 file implementation khớp nguồn. Profile RTX 5080
chỉ thay launch matrix apply. Không đổi assertion, tolerance hoặc công thức.

Bằng chứng đã audit: core 129 test đạt; lượt GeomLoss 0.3.1 gồm 64 test đạt;
12 case FP64 độc lập đạt. Đây là các phạm vi kiểm tra riêng, không cộng số test
để gọi là một lượt full mới. Máy local có Torch CPU, không có CUDA/Triton.

## 1. Làm rõ ba test OTT-Hessian đang skip

Tìm repo/commit dependency đúng API test tác giả yêu cầu. Ghim và kiểm tra hash
nguồn ngoài trước/sau chạy, thử lại test gốc bằng dependency này. Đặc biệt kiểm
tra `HessianALineax`. Nếu API không có trong nguồn public đã khảo sát, ghi rõ
coverage gap; không đổi sang `HessianA` hoặc tự viết adapter rồi gọi là test gốc.

Tiêu chí: có kết quả từng test (pass/fail/skip), traceback/reason, nguồn và
version dependency. Skip hoặc lỗi baseline bên ngoài không được tính là pass.

## 2. Kiểm tra cảnh báo early stopping

Tái tạo đúng fixture `test_flashstyle_symmetric_early_stopping`: CUDA RNG seed
0, x/y FP16, weights FP32, shape 128x128x32, eps 0.1, threshold 1e-3,
check_every 5, budget gốc 200. Đo marginal L1 từ plan dựng lại ở CPU FP64;
so với reference log-domain độc lập trên cùng input làm tròn. Thử budget
1000/4000/16000 khi cần, giữ nguyên threshold và các flags của lượt gốc.
Lưu input thực tế để có thể kiểm tra lại CUDA RNG và reference.

Tiêu chí: tách cờ dừng theo thay đổi potentials khỏi chứng nhận marginal;
reference phải tự đạt residual trước khi dùng đánh giá parity. Nếu chưa đạt
tại cap, ghi chưa xác nhận; không nới tolerance để gọi là hội tụ.

## 3. Chạy lại toàn bộ 20 file trong môi trường thống nhất

Dùng GeomLoss 0.3.1, profile RTX 5080 và cap CG đã xác nhận; chạy 12 case độc
lập. Audit JUnit, warnings, source/profile hashes, versions và ZIP checksum.
Nếu còn skip thì ghi `passed_with_coverage_gaps`, không gọi là full hoàn tất.

## 4. Mở rộng correctness tới workload benchmark

Thử nhiều shape, dimension, epsilon, weights; kiểm tra forward, gradient,
apply-plan và HVP. Đối chiếu bản cũ với bản tác giả trên cùng input, precision,
cost, schedule và stopping rule để xác định khác biệt cụ thể.

## 5. Chạy lại tám benchmark hiệu năng

Kiểm soát tải GPU, warmup, synchronization, precision và phép đo memory.
Tách phép so cùng số vòng khỏi phép so cùng ngưỡng hội tụ. HVP phải kèm CG
diagnostics. Ghi đúng nhãn author code với profile tương thích RTX 5080.

## 6. Chốt phiên bản làm nền FlashOPW

Ghim commit, dependency, patch launch và cấu hình đã kiểm chứng; lưu báo cáo
correctness và benchmark. Sau đó nối FlashOPW vào lõi này và kiểm tra lại các
thí nghiệm liên quan. Không kết luận kết quả cũ sai chỉ từ khác biệt runtime.

## Tiến độ bước 1–2 (09/10/2026)

- Đã khảo sát OTT-Hessian public tại commit
  `7eb189fe39982f587da935044480655b65939637`, toàn bộ 34 commit reachable từ
  các branch đã fetch, không phải shallow clone. Không tìm thấy chuỗi
  `HessianALineax` trong lịch sử này; hai test JAX vẫn thiếu API gốc.
- `torch_sinkhorn_hessian.py` có các tên API mà test KeOps yêu cầu. Chuẩn bị
  checkout riêng, ghim SHA-256 và lệnh thử GPU; chưa có kết quả GPU cho test này.
- Fixture early stopping chỉ assert số cập nhật <=205, hữu hạn và gần lời gọi
  fixed 200 vòng. Điều này chưa bắt buộc dừng sớm hoặc chứng nhận marginal.
- Chuẩn bị runner bước 1–2 và phép đo FP64. Kết quả GPU/residual của fixture
  vẫn chờ server; không coi việc viết runner là đã hoàn tất kiểm chứng.
- Kiểm tra local: 48 test đạt (reference FP64, phát hiện source/dependency bị
  sửa, accounting và runner); 131 file tác giả vẫn verified. 22 hash dependency
  được đối chiếu trực tiếp với Git blobs tại pin, không chỉ với working tree.

Nguồn public: [OTT-Hessian](https://github.com/yexf308/OTT-Hessian),
[test HVP tác giả tại pin](https://github.com/ot-triton-lab/flash-sinkhorn/blob/bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1/torch-ext/flash_sinkhorn/testing/test_hvp_parity.py).

Artifact cần xem từ server: commit/push ZIP, pull về audit, giữ bản local rồi
gỡ ZIP khỏi repo. File vẫn tồn tại trong lịch sử Git sau khi gỡ khỏi main.

## Thực thi bước 1–2 trên server

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 outputs/venv-author-flashsinkhorn/bin/python -u scripts/run_author_steps12.py
```

Runner tải OTT-Hessian vào `outputs/author-dependencies/OTT-Hessian`, checkout
pin và kiểm tra 22 file với SHA-256 đã đối chiếu Git blobs. Nếu thư mục đã có,
chỉ kiểm tra và từ chối nội dung khác pin; không reset hoặc ghi đè. Không cài
lại package, không đặt source ngoài vào hai cây FlashSinkhorn được bảo vệ.
Validator thêm đường dẫn import đã kiểm tra; test HVP vẫn là file tác giả gốc.
Plugin quan sát lưu backend thực dùng và CG diagnostics, không tăng cap 50 của
test này. Nếu baseline fallback từ KeOps sang dense thì không tính là pass KeOps.

Sau đó runner chạy chẩn đoán early stopping độc lập ngay cả khi baseline HVP
báo lỗi. Lưu input JSON chính xác từ CUDA, SHA-256 input, residual từng budget,
sai số plan với FP64 đã kiểm tra hội tụ, so FP64 cùng lịch 200 vòng, warnings,
versions và source/profile/dependency hashes. Mọi kernel/test gốc giữ nguyên.
Reference CPU FP64 có cap 64000, tolerance marginal 1e-6. Tiêu chí chẩn đoán
GPU marginal 1e-3 và plan relative L2 5e-3 được ghi riêng, không thay tolerance
upstream. `potential_stop_confirmed` và `marginal_confirmed` là hai cờ riêng.
Mỗi budget có cả lời gọi threshold gốc và control fixed-budget không dừng sớm:
nếu threshold đã dừng nhưng marginal chưa đạt, chỉ tăng cap không bảo đảm thêm
cập nhật. Control giúp phân biệt trường hợp đó với thiếu budget. Nếu control
đạt nhưng early stop chưa được chứng nhận, ghi
`fixed_budget_confirmed_stop_not_certified`, không gọi early stopping đã đạt.

Có heartbeat 30 giây khi process con còn chạy; reference FP64 ghi tiến độ mỗi
1000 vòng. Kết thúc tạo `outputs/author_flashsinkhorn_steps12_<timestamp>_<pid>.zip`
cùng manifest SHA-256. Các trạng thái:

- `steps12_confirmed` (exit 0): HVP không lỗi/skip và phép đo marginal/reference
  được xác nhận. Vẫn chỉ là bước 1–2, không thay lượt full bước 3.
- `completed_with_coverage_gaps` (exit 3): HVP còn skip hoặc chỉ control fixed
  budget đạt chứng nhận marginal/reference mà early stop chưa đạt.
- `failed_or_incomplete` (exit 1): lỗi test, nguồn sai hash hoặc phép đo hội tụ
  chưa được xác nhận. Đọc kết quả riêng từng bước, không coi bước 2 thất bại chỉ
  vì baseline HVP lỗi; ZIP vẫn được tạo để chẩn đoán.

Kết quả khảo sát API ở local và unit tests không thay kết quả GPU còn đang chờ.

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
- Artifact bước 1–2 nhận qua `dfb4166` đã được audit: test KeOps lỗi trong
  apply của OTT-Hessian (`ni: 16 và 256`) trước phép so HVP; hai test JAX
  skip do thiếu API. Bước 1 đã xác định nguyên nhân, chưa có parity HVP ngoài đạt.
- Fixture early stopping chỉ assert số cập nhật <=205, hữu hạn và gần lời gọi
  fixed 200 vòng. Điều này chưa bắt buộc dừng sớm hoặc chứng nhận marginal.
- Bước 2 đạt trên fixture gốc: threshold 1e-3 dừng sau 11382 cập nhật,
  marginal L1 `1.14732538e-4`, plan relative L2 `2.97348140e-3` so FP64 đã
  hội tụ. Budget 200 chưa đủ; cùng lịch cập nhật vẫn khớp FP64 với relative L2
  `2.44562416e-5`. Không đổi tolerance upstream.
- Chi tiết checksum, traceback, bảng budget và giới hạn bằng chứng nằm ở
  [báo cáo audit bước 1–2](../reports/author_flashsinkhorn_steps12_audit_20261009.md).
  ZIP đã sao lưu local và gỡ khỏi main theo quy trình chuyển file. Trạng thái
  tổng vẫn `failed_or_incomplete` do bước 1. Bước 3 đã chạy và audit lượt full:
  20 file, 401 passed, 1 failed, 2 skipped; trạng thái `failed` do OTT-Hessian.
  12 case FP64 độc lập đạt. Bước 4 đã chuẩn bị runner; chưa có kết quả GPU.
  Bước 5–6 chưa thực hiện.
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

Kết quả GPU bước 1–2 đã được audit trong báo cáo liên kết ở trên. Lệnh này
dùng để tái chạy; nó không sửa lỗi baseline hoặc bổ sung API JAX còn thiếu.

## Thực thi bước 3: một lượt full trong môi trường thống nhất

Runner chạy toàn bộ 20 file từ inventory ghim, mỗi file một process để giải
phóng VRAM, cùng interpreter/environment. Trước/sau kiểm tra GeomLoss 0.3.1
và hash checkout OTT-Hessian bước 1; truyền dependency này cho file HVP gốc.
Không cài package hoặc sửa dependency. Core dùng profile RTX 5080, cap CG 256
chỉ cho fixture double-backward đã kiểm chứng; 12 case FP64 độc lập vẫn bắt buộc.
Cảnh báo early stopping ở budget gốc vẫn được lưu, kết quả chẩn đoán bước 2
không thay budget test upstream.

Trên server GPU 1:

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only && env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 outputs/venv-author-flashsinkhorn/bin/python -u scripts/run_author_extended_validation.py --cg-cap 256 --require-geomloss-version 0.3.1 --ott-hessian-root outputs/author-dependencies/OTT-Hessian --output "outputs/author_flashsinkhorn_step3_$(date +%Y%m%d_%H%M%S)_$$"
```

Nếu checkout dependency bị thiếu/sai hash hoặc phiên bản GeomLoss khác yêu
cầu, runner dừng trước phép đo GPU và lưu lỗi. Không bỏ option để né preflight.
Nếu nguồn/môi trường hợp lệ, lỗi test HVP baseline không chặn chạy các file
còn lại. Lỗi layout KeOps và hai skip thiếu API JAX đã ghi ở bước 1 có thể
tiếp tục xuất hiện. Khi đó trạng thái tổng phải là `failed`, không chuyển
thành `passed_with_coverage_gaps` vì ngoài skip còn có test lỗi. Đánh giá từng
file và số test thực tế sau audit, không dự đoán số pass từ các lượt trước.

Kết thúc có ZIP `outputs/author_flashsinkhorn_step3_<timestamp>_<pid>.zip`.
Để chuyển ZIP mới nhất thuộc bước 3 bằng quy trình Git đã thống nhất:

```bash
(
set -e
cd /home/doanpt/minh.nd/flash-opw
artifact=$(outputs/venv-author-flashsinkhorn/bin/python -c 'from pathlib import Path; p=list(Path("outputs").glob("author_flashsinkhorn_step3_*.zip")); assert p, "No step 3 ZIP found"; print(max(p, key=lambda x:x.stat().st_mtime).as_posix())')
git add -f -- "$artifact"
git commit --only -m "Upload author FlashSinkhorn step 3 GPU evidence" -- "$artifact"
git push origin main
)
```

Sau khi user push: pull ZIP, audit checksum/JUnit/source/profile/dependency,
CG fixture và 12 case độc lập; sao lưu local rồi gỡ ZIP khỏi main và push
cleanup. Không gộp lượt này với các lượt subset cũ. Lệnh audit local:

```powershell
.venv-baselines/Scripts/python.exe scripts/audit_author_extended_validation.py outputs/author_flashsinkhorn_step3_<timestamp>_<pid>.zip
```

Audit chấp nhận bằng chứng một lượt test có lỗi nếu accounting/nguồn đúng,
nhưng vẫn trả trạng thái test `failed`; `failed_audit` là bằng chứng không đủ
hoặc không khớp. Bước 3 chưa thể gọi là full pass khi còn lỗi/skip baseline.

### Kết quả bước 3 đã audit (09/10/2026)

Artifact qua commit `9bd2535` đã chạy đủ 20 file, **401 passed, 1 failed,
2 skipped, 0 errors**. 19 file ngoài HVP parity đạt toàn bộ assertion;
lỗi KeOps và hai skip API JAX vẫn nằm trong `test_hvp_parity.py` như bước 1.
Không đổi trạng thái tổng `failed` thành pass. CG fixture hai đường hội tụ
sau 149 bước, residual `5.4462555e-7`; 12/12 case FP64 độc lập đạt.
Checksum, source/profile/dependency và phiên bản môi trường 20 file khớp.
Warnings early stopping còn được lưu, không suy ra hội tụ chỉ từ test pass.

Đã sao lưu local và gỡ ZIP chuyển giao khỏi main. Chi tiết từng file, metrics,
warnings và giới hạn nằm trong
[báo cáo audit bước 3](../reports/author_flashsinkhorn_step3_audit_20261009.md).
Bước tiếp theo theo kế hoạch là mở rộng correctness tới workload benchmark
(bước 4), giữ riêng phần đối chứng OTT-Hessian còn thiếu; chưa có kết quả GPU bước 4–6.

## Thực thi bước 4: correctness mở rộng và đối chiếu bản cũ

Đã chuẩn bị `scripts/run_author_step4.py`, chưa chạy GPU tại máy local.
Inventory cố định gồm **111 trường hợp**, chạy tuần tự, mỗi trường hợp trong
một process riêng để giải phóng VRAM:

- 56 case dense: 14 shape, hai schedule symmetric/alternating, hai precision
  IEEE FP32/TF32. Có dimension 1–1024, shape không vuông tới 512x640x13,
  epsilon 0.01/0.1/1, cost scale 1/0.5 và uniform/nonuniform weights. Epsilon,
  scale và weights phân bố theo shape, không phải tích Descartes đầy đủ.
- 38 case forward: 19 shape duy nhất từ các sweep benchmark forward và memory,
  hai schedule, TF32; gồm n tới 50000 và d tới 1024.
- 17 case HVP: toàn bộ shape duy nhất của hai sweep HVP, symmetric, IEEE FP32;
  n tới 50000, d tới 512.

Mỗi phép so dùng cùng input FP32, epsilon, cost scale và precision. Input được
sinh bằng CPU RNG rồi chuyển GPU, lưu SHA-256 để tái tạo khi audit; seed 0
không đồng nghĩa tensor trùng với benchmark cũ sinh bằng CUDA RNG. Các case
dense dùng U[0,1]/sqrt(d); shape benchmark dùng U[0,1]. Không sửa 131 file
tác giả hoặc source solver cũ. Dùng profile matrix-apply RTX 5080 đã kiểm tra;
autotune tắt trong lượt correctness. Đây chưa phải benchmark thời gian bước 5.

### Quy ước đối chiếu

Symmetric tác giả có thêm một cập nhật full ở đầu và một ở cuối; bản cũ native
chỉ có các cập nhật half-step. Runner lưu cả hai kết quả native, rồi dùng
adapter validation gọi kernel cũ với đúng các cập nhật đầu/cuối để so cùng
lịch. Không đổi solver cũ hoặc gọi adapter là bản cũ nguyên trạng. Alternating
dùng native hai bên. Fixed 10 vòng tương ứng 12 cập nhật symmetric hoặc 10
cập nhật alternating; không dùng early stopping cho phép so này.

Apply-plan hai chiều dùng cùng potentials tác giả và matrix giá trị x−0.5/y−0.5
(có phần tử âm), cùng vector ones. Gradient tác giả chuẩn hóa theo marginal
đích a/b, còn gradient bản cũ dùng marginal thực của P. Khi P chưa hội tụ, hai
định nghĩa có thể khác; mỗi bên được kiểm tra với công thức FP64 tương ứng và
lưu riêng chênh lệch quy ước. HVP hai bên dùng cùng P, IEEE, không preconditioner;
đơn vị damping được khớp bằng `legacy damping = epsilon * author tau2`, tau2=1e-5.
Đây là implicit operator có damping tại P đã cho; chỉ diễn giải thành Hessian
tại nghiệm OT khi forward đã được xác nhận hội tụ.

Dense chạy thêm 500 vòng, reference CPU FP64 độc lập cùng lịch; kiểm tra toàn
bộ P, apply và gradient. HVP so với hệ KKT FP64 giải trực tiếp, cap CG 256.
Shape benchmark HVP solve 100 vòng, cap CG 50; forward benchmark giữ 10 vòng.
Ở shape lớn chỉ kiểm tra 8 hàng và 8 cột trải đều, mỗi hàng/cột vẫn dùng toàn
bộ phía đối diện. Lưu full potentials, các output được chọn, input hashes,
CG diagnostics, warnings, versions và source/profile hashes trong ZIP.

Ngưỡng diagnostic được khai báo trước trong `author_step4_reference.py`:
relative L2 5e-4 cho IEEE, 5e-3 cho TF32, 5e-4 cho HVP; residual hệ KKT <=1e-9.
Đây là tiêu chí của bước 4, không sửa tolerance test upstream. Dense chỉ xác
nhận forward khi marginal L1 GPU dựng lại bằng FP64 <=1e-3 và reference <=1e-6.
Chưa đạt ở budget cố định phải ghi coverage gap. Shape lớn có marginal toàn
bộ đo bằng GPU FP32 để chẩn đoán, nhưng không có chứng nhận marginal toàn bộ
bằng FP64 hoặc reference solve đầy đủ. HVP chỉ tính parity đạt nếu cả hai
CG xác nhận residual <=max(1e-6, 1e-6*initial residual); nếu chưa đạt tại cap,
ghi số đo như diagnostic và coverage gap, không tính là HVP pass.

### Chạy trên server

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only && env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 outputs/venv-author-flashsinkhorn/bin/python -u scripts/run_author_step4.py
```

Dùng nguyên venv thống nhất của bước 3, GeomLoss 0.3.1; không cài lại package.
Runner in số case và heartbeat mỗi 30 giây, kiểm tra hash trước/sau và sau mỗi
process. Lỗi số học vẫn được lưu và runner chạy tiếp; thiếu report hoặc OOM
thì dừng, ghi rõ case chưa chạy. ZIP vẫn được tạo khi có lỗi trong lượt chạy.

Trạng thái tổng:

- `passed_checks` (exit 0): mọi assertion trong phạm vi đều đạt, không coverage gap.
- `passed_checks_with_coverage_gaps` (exit 3): mọi assertion đã thực hiện đạt,
  còn giới hạn chứng nhận hội tụ/CG hoặc lấy mẫu. Với inventory có shape lớn,
  đây là trạng thái dự kiến nếu không có assertion lỗi; không gọi full correctness pass.
- `failed`/`interrupted` (exit 1): có assertion lỗi, lỗi thực thi/integrity,
  hoặc người dùng dừng. Không bỏ case lỗi để đổi trạng thái thành pass.

Lỗi/skip OTT-Hessian của bước 3 vẫn là vấn đề riêng chưa được xử lý bởi runner này.

### Chuyển và audit artifact

Sau khi chạy xong, commit/push ZIP bước 4 mới nhất:

```bash
(
set -e
cd /home/doanpt/minh.nd/flash-opw
artifact=$(outputs/venv-author-flashsinkhorn/bin/python -c 'from pathlib import Path; p=list(Path("outputs").glob("author_flashsinkhorn_step4_*.zip")); assert p, "No step 4 ZIP found"; print(max(p, key=lambda x:x.stat().st_mtime).as_posix())')
git add -f -- "$artifact"
git commit --only -m "Upload author FlashSinkhorn step 4 GPU evidence" -- "$artifact"
git push origin main
)
```

Sau khi user push: pull, audit SHA-256 và source/profile/legacy/validator hashes,
tái sinh input theo hash, tính lại sai số từ arrays lưu; chạy lại reference
CPU FP64 và hệ KKT cho dense. Không chạy code trong ZIP. Auditor không thay
cho chạy lại CUDA và không chứng nhận toàn bộ output shape lớn chỉ từ lấy mẫu.
Sao lưu ZIP local rồi gỡ ZIP khỏi main và push cleanup theo quy trình đã thống nhất.

```powershell
.venv-baselines/Scripts/python.exe scripts/audit_author_step4.py outputs/author_flashsinkhorn_step4_<timestamp>_<pid>.zip --output outputs/step4-audit.json
```

Kiểm tra local đã bao gồm đối chiếu lịch cập nhật bằng CPU, đơn vị damping,
phát hiện arrays/tolerance bị sửa và runner giữ case lỗi khi tổng hợp/bundle.
Chưa có kết quả GPU bước 4 để ghi vào báo cáo thực nghiệm.

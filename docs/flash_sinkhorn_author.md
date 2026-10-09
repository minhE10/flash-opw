# FlashSinkhorn theo implementation chính thức

Ngày bắt đầu: **08/10/2026**. Repo nguồn:
<https://github.com/ot-triton-lab/flash-sinkhorn>.
Commit cố định: **`bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1`**,
package version **0.4.1**. Đây là HEAD được tải ngày 08/10, không phải tag
`v0.4.1` (tag đó trỏ tới một commit khác). Mọi phép đối chiếu dùng commit này.

## Hai thư mục riêng

| Thư mục | Vai trò |
|---|---|
| `external/flash-sinkhorn-upstream/` | Git clone của tác giả, detached tại commit cố định; chỉ đọc |
| `flash_sinkhorn_author/` | Bản implementation riêng: nguyên 131 file được Git theo dõi ở commit nguồn |

Bản reference không nằm trong Git của dự án này; lệnh `prepare` tải lại đúng
commit trên server hoặc máy khác. Bản implementation nằm trong Git của dự án,
giữ nguyên README, LICENSE MIT, package, tests và các ví dụ của tác giả.
`flash_sinkhorn_author/UPSTREAM.json` lưu SHA-256 và Git blob ID cho từng file.
Mọi file nguồn hiện khớp từng byte với Git commit của tác giả.

Trên Windows, reference có thuộc tính ReadOnly cho các file, kể cả `.git`.
Đây là thuộc tính file, không phải ACL cấm chủ sở hữu thay đổi quyền.
Trên Linux, `prepare` bỏ quyền ghi của cả file và thư mục. Công cụ `verify`
kiểm tra nội dung, commit và chế độ chỉ đọc; không reset/pull/ghi đè reference
đã có. Các output, cache pytest và bytecode không được ghi vào reference.

## Phạm vi implementation

Không dịch lại công thức hay thay kernel: bản đầu tiên sử dụng nguyên code
của tác giả. Những phần đã đưa vào gồm:

- Solver alternating và symmetric, shifted potentials, epsilon schedule,
  final extrapolation và stopping criterion của tác giả.
- Kernel LSE/fused symmetric, padding, cấu hình tile và autotuning.
- Apply-plan vector/matrix, gradients, autograd, CG và HVP.
- `SamplesLoss` và các kiểm tra half-cost, TF32, batching của tác giả.
- Bộ reference và test nguyên bản; không nới tolerance để làm test qua.

Repo 0.4.1 cũng có multiscale/FlashSinkhorn 2. Nguồn này được giữ nguyên trong
bản sao đầy đủ; validation `core` tập trung vào dense streaming FlashSinkhorn.
Validation `full` chạy toàn bộ bộ test trong `torch-ext/flash_sinkhorn/testing`.
Các test cần GeomLoss/KeOps/JAX hoặc repo OTT-Hessian ngoài có thể skip nếu thiếu
dependency; số skip được ghi riêng, không coi là đã kiểm chứng những backend đó.

API mới là **`flash_sinkhorn`** (có dấu gạch dưới). API implementation cũ là
`flashsinkhorn`. Runner mới ép đường dẫn import và xác nhận file thực tế được
import. Code FlashOPW và runner cũ chưa được chuyển sang lõi mới; các kết quả
đã lưu vẫn thuộc implementation cũ.

Một khác biệt cần kiểm tra khi chuyển FlashOPW là quy ước potentials. Code
tác giả dùng quy ước GeomLoss với trọng số trong LSE, và có bước chuyển sang
quy ước OTT cho HVP. Không đưa potentials của API này trực tiếp vào công thức
của API cũ mà chưa kiểm chứng (P), score, schedule và constant.
`SamplesLoss` còn có centering, epsilon scaling, debias và precision controls;
phải đặt chúng tường minh khi đối chiếu một bài toán fixed-iteration.

## Kiểm tra source trên Windows/CPU

Chạy tại thư mục gốc `flash-opw`:

```powershell
.venv-baselines/Scripts/python.exe scripts/author_flashsinkhorn_sources.py prepare
.venv-baselines/Scripts/python.exe scripts/validate_author_flashsinkhorn.py
.venv-baselines/Scripts/python.exe -m pytest -q tests/test_author_flashsinkhorn_sources.py
```

`prepare` đã được thực hiện local. Kiểm tra source và parse Python không cần
Torch/Triton và không xác nhận kernel CUDA chạy đúng. Kết quả được ghi ở
`outputs/author_flashsinkhorn_validation/validation.json`.

## Validation CUDA trên server

Dùng một Python environment riêng để tránh hai distribution cùng tên
`flash-sinkhorn` ghi đè nhau. Ví dụ sau kế thừa Torch/Triton từ conda environment
`minh`, và chỉ cài source mới vào venv riêng; không nâng cấp Torch hay driver.

```bash
cd /home/doanpt/minh.nd/flash-opw
conda activate minh

python scripts/author_flashsinkhorn_sources.py prepare
python -m venv --system-site-packages outputs/venv-author-flashsinkhorn
outputs/venv-author-flashsinkhorn/bin/python -m pip install \
  --no-deps --no-build-isolation -e ./flash_sinkhorn_author
outputs/venv-author-flashsinkhorn/bin/python -m pip show torch triton pytest

# Sau khi GPU 1 được cấp cho lượt chạy này:
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  outputs/venv-author-flashsinkhorn/bin/python scripts/validate_author_flashsinkhorn.py \
  --gpu --suite core --output outputs/author_flashsinkhorn_core_20261008
```

Environment cần có pytest. Nếu thiếu, cài pytest trong venv riêng trên. GPU
runner yêu cầu CUDA/Triton thật và đúng một GPU visible; thiếu GPU thì exit 2,
không tạo kết quả GPU pass. Runner giới hạn PyTorch allocator ở 45% VRAM.
Không dùng chung GPU với lượt benchmark cần đo timing được kiểm soát tải.

Sau khi core đã qua, có thể chạy `--suite full` với **output directory mới**.
Runner in trạng thái kiểm tra source, parse Python, kiểm tra GPU và chạy tests.
`validation.json` ghi `phase` hiện tại; log pytest được flush sau mỗi dòng.
Kiểm tra môi trường (import Torch/Triton và CUDA) có timeout mặc định 180 giây,
có thể điều chỉnh bằng `--preflight-timeout`. Timeout này không áp dụng cho
thời gian compile/autotune hoặc chạy bộ test. Preflight cũng ghi VRAM còn trống;
giới hạn allocator 45% không đặt trước VRAM và không ngăn process khác dùng GPU.

Tất cả lệnh test giữ nguyên test/tolerance của tác giả. Mỗi lượt lưu metadata,
đường dẫn package, log pytest, JUnit XML, số pass/fail/skip và kiểm tra hash
trước/sau test. Không ghi đè artifact của một lượt GPU đã có.

## Trạng thái và bước tiếp theo

### Cấu hình tương thích RTX 5080 (09/10/2026)

Log server ghi 119 test đạt, 8 lỗi shared memory (101632 byte yêu cầu so với
101376 byte giới hạn) và 2 lỗi CUDA OOM. Chưa có traceback đầy đủ của 8 lỗi
shared memory để xác nhận kernel/candidate cụ thể. Cấu hình dưới nhắm vào
apply-plan matrix dùng trong HVP và hai test OOM. Sau đó người dùng cung cấp
log lượt `author_flashsinkhorn_rtx5080_core_20261009_113028`: **129 passed,
0 failed/error/skip, 6 warnings, 171.70 s**, trạng thái `passed_compatibility`.
Đây là kết quả từ log được cung cấp, chưa phải audit đầy đủ artifact của lượt
chạy. Có 5 cảnh báo API deprecated và 1 cảnh báo CG chưa hội tụ ở test
double-backward (residual 0.000133 so với ngưỡng 1e-6 tại 64 bước).

`--kernel-profile rtx5080` tạo bản sao riêng trong output của mỗi lượt, chỉ
thay launch controls tại `apply_plan_mat_flashstyle`: `block_m=block_n=32`,
`block_k=block_d=16`, `num_stages=1`, `autotune=False`. Thay tại hàm launcher
để mọi lần gọi qua HVP và re-export đều nhận cấu hình, kể cả các lần gọi
HVP không truyền block/stages từ bên ngoài. Cấu hình này ghi đè launch controls
được truyền vào hàm apply matrix; `num_warps` giữ giá trị của caller.
Các kernel tính toán, dtype, TF32, exp/exp2, epsilon, damping, số vòng CG,
ngưỡng hội tụ và tolerance/test của tác giả giữ nguyên.

Cả `external/flash-sinkhorn-upstream/` và `flash_sinkhorn_author/` vẫn nguyên
byte. Bản sao được sinh ở `implementation-rtx5080/`; `kernel-profile.patch`
ghi diff và `kernel-profile.json` ghi cấu hình/commit/hash 131 file. Runner
kiểm tra cả bản gốc lẫn bản tương thích trước/sau test. Import probe xác nhận
package nằm trong bản sao tương thích, không rơi về editable install bản gốc.
Trạng thái đạt được ghi `passed_compatibility` (hoặc
`passed_with_skips_compatibility`), phân biệt với validation upstream.

Chạy 10 test vừa lỗi trước, rồi toàn bộ core nếu lượt đầu đạt:

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  outputs/venv-author-flashsinkhorn/bin/python scripts/validate_author_flashsinkhorn.py \
  --gpu --suite regressions --kernel-profile rtx5080 \
  --output "outputs/author_flashsinkhorn_rtx5080_regressions_$(date +%Y%m%d_%H%M%S)"

# Sau khi regressions đạt:
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  outputs/venv-author-flashsinkhorn/bin/python scripts/validate_author_flashsinkhorn.py \
  --gpu --suite core --kernel-profile rtx5080 \
  --output "outputs/author_flashsinkhorn_rtx5080_core_$(date +%Y%m%d_%H%M%S)"
```

Không cần cài lại package vì runner dùng đường dẫn bản sao cho từng lượt.
Mặc định `--kernel-profile upstream` tiếp tục chạy nguyên bản. Có thể kiểm tra
việc tạo bản tương thích trên CPU bằng cùng lệnh nhưng bỏ `--gpu`; trạng thái
lúc đó là `static_profile_verified_gpu_pending`, không phải CUDA pass.

Autotuning của apply matrix bị tắt nên các test `test_flashstyle_autotune_parity`
trong profile này so các launch cố định, không chứng minh autotuner đạt.
Các test vẫn dùng nguyên tolerance nhưng phạm vi coverage phải hiểu theo cấu
hình đang chạy. Profile này phục vụ correctness trên RTX 5080, chưa tối ưu
timing và không thay kết quả benchmark upstream. Process khác chiếm VRAM vẫn
có thể gây OOM; runner không dừng process đó hay nới tolerance để vượt lỗi.

### Kiểm tra hội tụ CG của HVP

Test `test_samplesloss_double_backward_matches_hvp_x_reference` đặt tường minh
`hvp_max_cg_iter=64` (API `SamplesLoss` mặc định 300). Test gốc chỉ so HVP
autograd với HVP gọi trực tiếp, không bắt buộc cờ hội tụ của hai phép giải.
Vì thế test có thể đạt kèm cảnh báo residual vượt ngưỡng.

Chế độ bổ sung `--hvp-max-cg-iter N` chỉ tăng budget của đúng case này, qua
pytest plugin ngoài cây tác giả. Bản test trên đĩa và assertion parity
`rtol=atol=1e-5` còn nguyên. Damping `tau2=1e-5`, CG `rtol=atol=1e-6`,
precision, epsilon, ba vòng forward và các setting khác giữ nguyên.
Metadata `cg_scenario` ghi rõ đây là test với budget thay đổi.

Plugin ghi cả hai `HvpInfo` vào `cg_convergence.json`, tính ngưỡng
`max(cg_atol, cg_rtol * cg_initial_residual)` theo code tác giả. Phải có
output hữu hạn, `cg_converged=True` và residual thực không vượt ngưỡng trên
cả autograd/reference, đồng thời assertion parity gốc phải đạt. Thiếu một
đường gọi hoặc còn chưa hội tụ làm validation thất bại. Không lọc cảnh báo.

Một lệnh thử lần lượt 128/256/512/1024 bước rồi chạy lại core tại budget đầu
tiên đạt cả hội tụ và parity:

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  bash scripts/run_author_hvp_cg.sh
```

Script giữ artifact từng lượt, chỉ tăng budget tiếp khi diagnostics xác nhận
CG đã chạm cap mà chưa đạt ngưỡng; lỗi source/CUDA/parity khi đã hội tụ sẽ
dừng để kiểm tra. Nếu không đạt tại 1024, script báo thất bại, không tự đổi
damping hay tolerance. Có thể chỉ chạy một budget:

```bash
CUDA_VISIBLE_DEVICES=1 outputs/venv-author-flashsinkhorn/bin/python \
  scripts/validate_author_flashsinkhorn.py --gpu --suite cg \
  --kernel-profile rtx5080 --hvp-max-cg-iter 256 \
  --output "outputs/author_flashsinkhorn_cg_256_$(date +%Y%m%d_%H%M%S)"
```

Core đạt với kiểm tra bổ sung có trạng thái `passed_compatibility_cg_checked`.
Chạy lại core không có `--hvp-max-cg-iter` vẫn giữ nguyên budget 64 của test
gốc và có thể còn cảnh báo. Kết quả này kiểm chứng hội tụ của fixture nêu
trên; benchmark HVP cần tự kiểm tra hội tụ trên từng workload thực tế.
Máy local không có CUDA; xác nhận GPU dưới đây dựa trên log người dùng cung cấp.

Log lượt `author_flashsinkhorn_rtx5080_core_cg_20261009_150850_3207450_256`
xác nhận **129 passed, 0 failed/error/skip, 5 warnings trong 87.55 s**;
trạng thái **`passed_compatibility_cg_checked`**. Budget được chọn trong lượt
sweep này là **256**; cả hai phép giải thực tế dừng sau **149 bước**:

| Đường HVP | Bước CG | Residual thực | Ngưỡng | Confirmed |
|---|---:|---:|---:|---|
| Autograd | 149 | 5.446255499919062e-7 | 1e-6 | True |
| Reference gọi trực tiếp | 149 | 5.446255499919062e-7 | 1e-6 | True |

Đã xác nhận hội tụ và parity của fixture này ở budget 256, với damping và
tolerance giữ nguyên. Reference ở đây dùng cùng solver HVP gọi trực tiếp,
không phải đối chứng Hessian độc lập. Chưa nhận đủ `validation.json`,
`cg_convergence.json` và JUnit XML để audit artifact độc lập. Thời gian trên
là tổng thời gian chạy test, không dùng làm benchmark speedup.

- Đã tải repo, ghim commit và đặt reference chỉ đọc.
- Đã tạo bản implementation đầy đủ khớp 131 file của tác giả.
- Đã bổ sung kiểm tra source, test phát hiện sửa file và runner CUDA.
- Kiểm tra local: 131 file khớp SHA-256/Git blob; 159 file reference (kể cả
  `.git`) có thuộc tính ReadOnly; 82 file Python parse thành công; 4 test
  kiểm tra source và việc từ chối GPU khi chưa được chọn đều đạt.
- **Chưa chạy validation CUDA trên máy Windows này:** Torch 2.11.0+cpu,
  không có Triton/CUDA.
- Chưa có benchmark mới hoặc kết quả FlashOPW dựa trên lõi mới.

Sau lượt core GPU đã đạt: đối chiếu chi tiết bản cũ với bản tác giả trên
cùng input/weights/cost/schedule, rồi xây adapter FlashOPW dùng lõi tác giả.
Kiểm tra lại correctness/hội tụ trước khi chạy các nhóm accuracy và timing.
Sai khác runtime so paper hoặc implementation khác repo chưa tự chứng minh
kết quả cũ sai; cần xác định sai khác bằng phép đối chiếu này.

### Validation mở rộng sau core/CG

Đã thêm workflow `scripts/run_author_extended_validation.py` để kiểm tra phần
còn lại của FlashSinkhorn author trước khi chuyển sang FlashOPW:

1. Kiểm tra nguồn ghim commit và tạo profile RTX 5080 trong output riêng.
2. Chạy **12 case độc lập**: ba shape `(17,23,3)`, `(32,24,16)`, `(37,29,33)`;
   symmetric/alternating, full/half squared cost, weights không đều. Input được
   tạo trên CPU rồi làm tròn FP32, reference dùng chính giá trị đó ở CPU FP64.
   So plan với một phép giải log-domain độc lập, kiểm tra marginal residual,
   apply matrix cả hai trục, và so HVP với phép giải trực tiếp hệ KKT FP64.
3. Chạy **toàn bộ 20 file test** của commit tác giả, mỗi file một process riêng.
   Điều này bao gồm chạy lại core để gói artifact mới có đầy đủ bằng chứng.
   Chỉ fixture double-backward đã xác nhận trước đó dùng cap 256 qua plugin;
   các tolerance, damping và test khác giữ nguyên.
4. Kiểm tra hash sau chạy, lưu environment, log, JSON/JUnit XML, diff/hash của
   profile, rồi tạo ZIP cùng manifest SHA-256. ZIP không chứa venv/cache/cây
   implementation lặp lại. Có heartbeat mỗi 30 giây khi process con còn chạy.

Reference mới nằm ngoài cây tác giả, tại `scripts/author_flashsinkhorn_fp64.py`.
Nó không import solver/reference/CG của tác giả. HVP được suy ra bằng cách vi
phân điều kiện marginal, giải hệ tuyến tính đầy đủ bằng `torch.linalg.solve`,
rồi tính đạo hàm của plan và gradient. Product đối chứng dùng cùng `tau2=1e-5`
với GPU; đây là HVP implicit có damping. Unit test riêng dùng `tau2=0`, so với
sai phân hữu hạn của gradient nghiệm OT đã hội tụ, đồng thời kiểm tra tuyến
tính/đối xứng của product có damping. HVP GPU được so tại chính potentials
OTT đã làm tròn mà kernel nhận; phép kiểm tra plan/hội tụ được thực hiện riêng.
Các case nhỏ này không thay validation toàn sweep hoặc benchmark hiệu năng.

Chạy trên server bằng môi trường author đã dùng thành công:

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  outputs/venv-author-flashsinkhorn/bin/python -u \
  scripts/run_author_extended_validation.py
```

Không cần cài lại Torch hoặc chạy lại sweep chọn CG. Script không tự cài thêm
dependency. Bộ full có test dùng GeomLoss, KeOps, JAX GPU/OTT-JAX và checkout
`3rd-party/OTT-Hessian`; các dependency thiếu có thể dẫn đến skip. Upstream yêu
cầu `geomloss>=0.3` trong dev extras và một test import
`geomloss._legacy.sinkhorn_samples`: phiên bản cũ có thể gây **collection error**,
không phải sai số kernel. Từng lỗi/skip và phiên bản package được giữ trong
artifact, không sửa test để bỏ qua. Profile chỉ thay matrix apply; bộ full vẫn
có thể phát hiện launch vượt tài nguyên ở kernel khác. Một số test unbalanced
dùng 20.000 vòng nên thời gian full có thể dài hơn nhiều so với core.

Trạng thái cuối:

- `passed_extended_compatibility` (exit 0): numerics đạt, cả 20 file hoàn tất,
  không failure/error/skip, source/profile kiểm tra đạt.
- `passed_with_coverage_gaps` (exit 3): phần đã chạy đạt nhưng còn skip; chưa
  được gọi là full validation. Module toàn skip ghi `not_validated_all_skipped`.
- `failed` (exit 1): có lỗi, numerics không đạt hoặc thiếu file chưa chạy.

Đường dẫn ZIP được in ở cuối. Sau khi đưa ZIP về máy local, có thể audit mà
không giải nén/chạy code trong artifact:

```powershell
.venv-baselines/Scripts/python.exe scripts/audit_author_extended_validation.py outputs/author_flashsinkhorn_extended_<timestamp>_<pid>.zip
```

Auditor tính lại checksum, số test từ XML, kiểm tra inventory theo commit,
metadata hash của profile, residual/tham số CG và metrics độc lập. Đây là kiểm
tra bằng chứng đã lưu, không phải chạy lại CUDA hay xác thực danh tính server.
Trạng thái full đạt vẫn phải đọc cùng warnings của từng file: assertion của
upstream không bắt buộc hội tụ của mọi lời gọi CG. Kiểm tra CG nghiêm ngặt mới
áp dụng cho fixture đã chỉ định và 12 case độc lập; không suy rộng ra mọi HVP.
**Chưa có kết quả GPU cho workflow mở rộng này trên máy local CPU.**

### Audit lượt mở rộng và kiểm tra lại GeomLoss

Artifact nhận qua commit `e34c467` đã được audit: **374 passed, 1 failed,
2 collection errors, 3 skipped**; checksum/inventory/metadata không có lỗi.
Cả **12 case FP64 độc lập đạt**, max HVP relative L2 khoảng `4.888e-7`.
Chi tiết và giới hạn nằm ở
[`author_flashsinkhorn_extended_audit_20261009.md`](../reports/author_flashsinkhorn_extended_audit_20261009.md).
ZIP đã được lưu local và gỡ khỏi main ở `3499c3f` theo quy trình chuyển file
server bằng Git do người dùng yêu cầu; lịch sử Git vẫn chứa ZIP.

Ba nonpass đều do GeomLoss 0.2.6 thiếu `geomloss._legacy`. Dev extras tác giả
yêu cầu `geomloss>=0.3`; wheel 0.3.1 đã được kiểm tra có các import cần thiết.
Chạy trên server để chỉ cài GeomLoss đã ghim hash vào venv author, rồi kiểm tra
lại năm file phụ thuộc package này và 12 case độc lập:

```bash
conda activate minh
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only
outputs/venv-author-flashsinkhorn/bin/python -m pip install \
  --no-deps --only-binary=:all: --require-hashes -r requirements-author-validation.txt
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  outputs/venv-author-flashsinkhorn/bin/python -u \
  scripts/run_author_extended_validation.py \
  --test-file test_autograd_semantics.py \
  --test-file test_geomloss_sinkhorn_triton.py \
  --test-file test_geomloss_vs_triton.py \
  --test-file test_half_cost.py \
  --test-file test_unbalanced_sinkhorn.py
```

Nếu lệnh cài package báo lỗi, dừng trước bước test. Lệnh `pip` trên không cài
lại Torch/Triton và không cài dependency bắc cầu. Năm file được chọn bao gồm
hai file từng đạt nhưng cũng import GeomLoss, vì dependency đã đổi phiên bản.
Không sửa kernel, source tác giả, damping hoặc tolerance trong bước này.

`--test-file` lặp được ở workflow mở rộng; bỏ option sẽ chạy toàn bộ 20 file
như trước. Lượt chọn file lưu `scope=subset` và inventory đầy đủ trong summary,
vẫn có heartbeat, kiểm tra hash, environment và ZIP SHA-256. Khi đạt hoàn toàn
trong phạm vi đã chọn, trạng thái là **`passed_subset_compatibility`** (exit 0),
không phải `passed_extended_compatibility`. Auditor phân biệt hai phạm vi và
chỉ yêu cầu CG fixture API nếu đã chọn file API; 12 case độc lập vẫn bắt buộc.
Ba skip OTT-Hessian của lượt trước chưa được xử lý bằng lượt GeomLoss này.

Artifact kiểm tra lại `author_flashsinkhorn_extended_20261009_174242_3217238`
đã nhận qua commit `c9d999c` và audit thành công: **64/64 test đạt** trên năm
file đã chọn, **12/12 case FP64 độc lập đạt**, mọi CG trong nhóm độc lập hội
tụ. Checksum, JUnit và metadata source/profile khớp; environment xác nhận
GeomLoss 0.3.1. Kết quả là `passed_subset_compatibility`, không phải một lượt
full mới. Còn một warning early stopping chưa xác nhận hội tụ trong 202 cập
nhật ở fixture tương ứng; ba skip OTT-Hessian vẫn chưa được kiểm chứng.
ZIP đã lưu local cùng audit rồi được gỡ khỏi repo theo quy trình chuyển file.
Chi tiết và SHA-256 nằm trong báo cáo audit được liên kết ở trên.

### Kế hoạch sáu bước và thực thi bước 1–2

Xem [kế hoạch kiểm tra](flash_sinkhorn_author_validation_plan.md) để theo dõi
sáu bước và tiêu chí hoàn tất. `scripts/run_author_steps12.py` chạy nguyên test
HVP với checkout OTT-Hessian public đã ghim hash, rồi đo marginal/FP64 cho đúng
fixture early stopping. Runner giữ nguyên API test, kernel và tolerance; không
alias `HessianALineax` sang hàm khác. Cả hai bước lưu chung một ZIP có checksum
để chuyển bằng quy trình Git đã thống nhất. Artifact bước 1–2 qua `dfb4166`
đã được [audit](../reports/author_flashsinkhorn_steps12_audit_20261009.md):
early stopping đạt tiêu chí marginal/reference sau 11382 cập nhật trên fixture;
test HVP KeOps lỗi layout tensor trong baseline OTT-Hessian, hai test JAX
thiếu API nên skip. Trạng thái tổng vẫn `failed_or_incomplete`; không có
parity HVP ngoài đạt từ lượt này. ZIP đã sao lưu local và gỡ khỏi main.

Bước 3 dùng `run_author_extended_validation.py` không chọn subset, thêm
`--require-geomloss-version 0.3.1` và `--ott-hessian-root` trỏ checkout đã
ghim ở bước 1. Runner và auditor ghi/kiểm tra dependency trước/sau, giữ lỗi
baseline thật trong kết quả, chạy tiếp các file còn lại. Lệnh đầy đủ và cách
chuyển ZIP nằm trong [kế hoạch bước 3](flash_sinkhorn_author_validation_plan.md#thực-thi-bước-3-một-lượt-full-trong-môi-trường-thống-nhất).
Lượt full bước 3 qua `9bd2535` đã được
[audit](../reports/author_flashsinkhorn_step3_audit_20261009.md): đủ 20 file,
401 passed, 1 failed, 2 skipped, không collection error. 19 file ngoài HVP
parity đạt; file HVP còn lỗi baseline KeOps và thiếu API JAX. CG fixture
hội tụ cả hai đường và 12/12 case độc lập đạt. Trạng thái tổng vẫn `failed`.
ZIP đã sao lưu local và gỡ khỏi main; bước 4–6 chưa thực hiện.

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

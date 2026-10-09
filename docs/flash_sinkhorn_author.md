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

- Đã tải repo, ghim commit và đặt reference chỉ đọc.
- Đã tạo bản implementation đầy đủ khớp 131 file của tác giả.
- Đã bổ sung kiểm tra source, test phát hiện sửa file và runner CUDA.
- Kiểm tra local: 131 file khớp SHA-256/Git blob; 159 file reference (kể cả
  `.git`) có thuộc tính ReadOnly; 82 file Python parse thành công; 4 test
  kiểm tra source và việc từ chối GPU khi chưa được chọn đều đạt.
- **Chưa chạy validation CUDA trên máy Windows này:** Torch 2.11.0+cpu,
  không có Triton/CUDA.
- Chưa có benchmark mới hoặc kết quả FlashOPW dựa trên lõi mới.

Sau khi validation CUDA đạt: đối chiếu chi tiết bản cũ với bản tác giả trên
cùng input/weights/cost/schedule, rồi xây adapter FlashOPW dùng lõi tác giả.
Kiểm tra lại correctness/hội tụ trước khi chạy các nhóm accuracy và timing.
Sai khác runtime so paper hoặc implementation khác repo chưa tự chứng minh
kết quả cũ sai; cần xác định sai khác bằng phép đối chiếu này.

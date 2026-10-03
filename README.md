# FlashSinkhorn in Triton + toy experiments

Implementation độc lập của thuật toán FlashSinkhorn trong
[paper của Ye et al.](https://arxiv.org/abs/2602.03067), với Sinkhorn log-domain
chuẩn làm đối chứng. Kernel Triton tính theo tile, không tạo ma trận chi phí
hay transport `n × m` trong quá trình giải. Xem [công thức và phạm vi](docs/method.md).

Môi trường đích do người dùng cung cấp: Ubuntu, RTX 5080 **16 GB/GPU**,
driver 570.172.08, PyTorch **2.11.0+cu128**, CUDA runtime 12.8. Mỗi lần chạy
chỉ dùng **một GPU được cấp quyền**. Ba card không tự hợp thành 48 GB cho một job.
Không thay driver, CUDA hệ thống hay môi trường của người khác.

## Chạy trên server

Trong môi trường riêng đã tạo, clone lần đầu vào thư mục riêng:

```bash
git clone https://github.com/minhE10/flash-opw.git
cd flash-opw
```

Nếu đã clone, cập nhật khi không có job đang chạy từ checkout này:

```bash
git pull --ff-only origin main
```

Kiểm tra và cài các thành phần còn thiếu **trong môi trường riêng**:

```bash
python -c "import sys, torch, triton; print(sys.executable); print(torch.__version__, torch.version.cuda, triton.__version__)"
python -m pip install -e '.[dev,plots]'
```

Lệnh cài không yêu cầu nâng cấp PyTorch; không thêm `--upgrade`. Triton nên là
phiên bản đi kèm bản PyTorch Linux đã cài. Nếu `import triton` lỗi, gửi traceback
và `python -m pip show torch triton` để kiểm tra môi trường trước khi đổi phiên bản.
`nvidia-smi` hiển thị mức CUDA driver hỗ trợ; `torch.version.cuda` là runtime
của bản PyTorch. Hai thông tin này không xác nhận một CUDA Toolkit hệ thống đã cài.

**Chỉ sau khi được phân GPU**, ví dụ GPU 1:

```bash
bash scripts/run_server.sh 1 --sizes 128 256 512 --weighted --target-ratio 1.3
```

Script lần lượt chạy smoke test, toàn bộ test bắt buộc GPU và benchmark.
Nếu bất kỳ bước nào lỗi, script dừng. `CUDA_VISIBLE_DEVICES=1` ánh xạ GPU vật lý
được chọn thành `cuda:0` bên trong chương trình. GPU đang trống trong `nvidia-smi`
**không phải** là quyền sử dụng hay cơ chế giữ chỗ.

Nếu dùng Slurm/scheduler, nhận allocation theo quy định server, giữ nguyên
`CUDA_VISIBLE_DEVICES` của scheduler, rồi chạy trực tiếp trong job:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python -m experiments.smoke
python -m pytest -q --require-gpu
python -m experiments.toy --device cuda --sizes 128 256 512 --weighted --target-ratio 1.3
```

Ngoài scheduler, cũng có thể dùng các lệnh này sau khi tự đặt
`export CUDA_VISIBLE_DEVICES=<GPU được cấp>`. CLI từ chối chạy CUDA nếu chưa
chọn đúng một GPU. Không tự dò GPU rảnh, không kill/reset GPU, không dùng DDP.
Mặc định giới hạn allocator PyTorch ở 25% VRAM (~4 GB trên một RTX 5080),
2 CPU threads, dữ liệu sinh trong RAM và không có DataLoader workers.
Giới hạn allocator không giữ chỗ GPU và không bao gồm mọi bộ nhớ driver/JIT;
các job cùng GPU vẫn có thể xung đột. Điều chỉnh giới hạn theo allocation:

```bash
python -m experiments.toy --device cuda --memory-fraction 0.20 --max-dense-mib 256
```

Benchmark lớn hơn, chỉ sau khi smoke/test đã qua và được phép sử dụng tài nguyên:

```bash
python -m experiments.toy --device cuda --datasets gaussian mixture \
  --dim 64 --sizes 256 512 1024 --iters 200 --repeats 3
```

## Kết quả và cách so sánh

Ba dataset tự sinh: Gaussian dịch chuyển, Gaussian mixture và hai vòng tròn.
Có seed cố định, trọng số không đều (`--weighted`), số điểm hai phía khác nhau
(`--target-ratio`), hai lịch cập nhật và lựa chọn precision.

Mỗi lần chạy tạo thư mục mới `outputs/<UTC timestamp>/`:

- `results.csv`: thời gian, peak allocation, speedup, sai số marginal, cost,
  mục tiêu entropy, dual và chênh lệch transport với baseline.
- `environment.json`: phiên bản, GPU, commit, trạng thái dirty và toàn bộ tham số.
- `timings.json`: từng lần đo, ngoài lần warmup/compile.
- `validation.json`: đạt/không đạt ngưỡng agreement và hội tụ.
- `comparison.png`, `transport.png`: biểu đồ so sánh và map barycentric 2D.

Thời gian đo là wall time đã đồng bộ CUDA, có setup/validation của solver,
không tính JIT warmup hay diagnostics. Hai solver nhận cùng tensor, epsilon,
số vòng lặp và lịch cập nhật. Peak memory là phần tăng của **PyTorch allocated**
so với bộ nhớ đầu vào đang sống, không phải tổng VRAM trong `nvidia-smi`.
Dữ liệu và plot không nằm trong vùng đo. Không dùng kết quả CPU để báo speedup Triton.

CLI trả exit code khác 0 khi sai số vượt ngưỡng; kết quả đã đo vẫn được giữ.
Nếu residual lớn, tăng `--iters` và chạy cả hai solver với cùng số vòng lặp.
`regularized_primal` có thể thuộc coupling chưa khả thi; `primal_minus_dual`
không phải chứng nhận optimality gap khi marginal chưa khớp.

## Kiểm thử local Windows/CPU

Không cần cài Triton trên Windows. Trong repo:

```powershell
python -m pytest -q
python -m experiments.toy --device cpu --sizes 32 65 --iters 100 --repeats 1 --weighted --target-ratio 1.3
```

Chế độ CPU so sánh Sinkhorn dense với **Torch tiled oracle** để xác nhận công
thức. GPU tests được đánh dấu skip nếu không có GPU được chọn. Trên server dùng
`--require-gpu` để không nhầm một lần chạy toàn skip thành xác nhận GPU.
Xem [nhật ký xác minh](docs/validation.md) để biết phần nào thực sự đã chạy.

Developer checks trên Linux/WSL có Triton và NumPy, không cần GPU:

```bash
python scripts/interpret_kernels.py
python scripts/compile_kernels.py --arch 120
```

Hai lệnh kiểm tra logic kernel qua CPU interpreter và biên dịch offline; không
phải kết quả thực thi hay benchmark GPU. `sm_120` là kiến trúc đích RTX 5080
([thông số NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)).

## Python API

```python
import torch
from flashopw import sinkhorn_flash, apply_plan, diagnostics, point_gradients

# Khởi động Python với CUDA_VISIBLE_DEVICES đã trỏ tới GPU được cấp.
x = torch.randn(256, 32, device="cuda", dtype=torch.float32) * 0.1
y = torch.randn(384, 32, device="cuda", dtype=torch.float32) * 0.1
result = sinkhorn_flash(x, y, epsilon=0.2, n_iters=300, tol=1e-4)
print(diagnostics(result))
py = apply_plan(result, y)  # P @ y, không tạo P
grad_x, grad_y = point_gradients(result)  # envelope gradient khi đã hội tụ
```

Đây là API forward + gradient tường minh, chưa tích hợp `.backward()`/HVP.
`materialize_plan` chỉ dành cho kiểm tra bài toán nhỏ và có giới hạn kích thước.
Không hỗ trợ unbalanced OT trong bản này dù tên thư mục cha có “FlashUOT”.

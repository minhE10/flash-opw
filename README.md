# FlashSinkhorn và FlashOPW

Chọn hyperparameter trên train và benchmark chuỗi dài: [hướng dẫn chạy](docs/opw_tuning_scaling.md).
Kế hoạch experiment 1–6: [file theo dõi](docs/opw_experiment_plan.md).
Nhóm2 — prior, score, Taylor: [protocol và lệnh chạy](docs/opw_group2.md).

FlashOPW áp dụng sửa cost trong **`main (2).pdf`**, với một tọa độ thời gian
bổ sung; loss mặc định theo phương trình (19). Lõi tự reproduce FlashSinkhorn
nằm riêng trong `flashsinkhorn`, API OPW nằm trong `flashopw`.
Xem [công thức, phạm vi và trial k-NN MAP/ACC](docs/flash_opw.md).
Kết quả kiểm tra local và giới hạn hiện tại ở [báo cáo validation](docs/validation_opw.md).

Trong môi trường `minh` trên server, sau khi đã được cấp GPU 1:

```bash
git pull --ff-only origin main
python -m pip install -e '.[dev,plots,opw]'
bash scripts/validate_flash_opw.sh 1
```

Khi validation qua, chạy pilot so sánh FlashOPW, affine dense và 10 metric
của journal trên cùng subset FacesUCR:

```bash
bash scripts/run_opw_knn.sh 1 --datasets FacesUCR \
  --max-train 32 --max-queries 16 \
  --output outputs/opw_knn_facesucr_pilot_20261006
```

Kết quả nằm trong `knn_results.csv`, `affine_parity.json` và `diagnostics.json`.
Đây là trial tham số cố định; chưa phải tái hiện toàn bộ bảng journal.

## FlashSinkhorn

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

## Phase 1: paper-style synthetic benchmark

After a single GPU has been allocated, run the reproducible Phase 1 preset:

```bash
bash scripts/run_phase1.sh 1
```

The preset uses a Gaussian point-cloud pair with `n=m=10000`, `d=64`,
`epsilon=0.1`, 10 Sinkhorn iterations, the symmetric schedule, TF32,
10 unmeasured warmups, and 50 measured repetitions. It is close to the
paper's forward benchmark while using a memory budget suitable for a 16 GB
RTX 5080. Results are written to `outputs/<UTC timestamp>/`.

For a quick validation run, override the measurement counts:

```bash
bash scripts/run_phase1.sh 1 --warmups 2 --repeats 3
```

Run this only on a GPU allocated to the job. The script does not discover,
reset, or take over another process's GPU.

## Phase 2: real-data benchmark

Install the optional torchvision dependency in the server environment:

```bash
python -m pip install -e '.[phase2]'
```

Then run the MNIST to Fashion-MNIST benchmark with ResNet18 penultimate-layer
features (`d=512`):

```bash
bash scripts/run_phase2.sh 1
```

The default sizes are `5000`, `10000`, `15000`, and `20000`, with
`epsilon=0.1`, 10 fixed iterations, TF32, 10 warmups, and 50 measured runs.
Features and downloaded datasets are cached under `data/`. Dense comparison is
automatically skipped when the estimated working set exceeds the configured
RTX 5080 memory budget; those rows are reported as Flash-only rather than
causing an out-of-memory run.

This phase uses the repository's supported squared-Euclidean cost on the
paper's real feature representation. It does not yet enable OTDD's additional
class-label lookup cost; that requires extending the solver API and Triton
kernel.

## Baseline comparison and paper-style illustrations

Install the optional comparison stack:

```bash
python -m pip install -e '.[baselines]'
python -c "import jax; print(jax.devices())"
```

Run the n- and d-sweeps on MNIST/Fashion-MNIST:

```bash
bash scripts/run_baselines.sh 1
```

This compares FlashSinkhorn with GeomLoss/KeOps, GeomLoss/Tensorized, and
OTT-JAX, then writes `paper_style_baselines.png` and
`baseline_results.csv`. The n-sweep uses the 512D ResNet18 features; the
d-sweep uses deterministic leading slices of those features. Tensorized rows
are skipped when their estimated quadratic working set exceeds the configured
VRAM budget. GeomLoss baselines use its legacy `SamplesLoss` API with
`debias=False`, `blur=sqrt(epsilon)`, and `scaling=0.9`; the exact backend and
status are recorded in the CSV.

For a short smoke benchmark:

```bash
bash scripts/run_baselines.sh 1 \
  --n-sizes 5000 --d-sizes 64 256 512 --d-sweep-n 5000 \
  --warmups 2 --repeats 3
```

The FlashSinkhorn engine lives in `flashsinkhorn`. The separate `flashopw`
package is reserved for the order-preserving cost and OPW solver; it is no
longer an alias for ordinary Sinkhorn. Existing `FLASHOPW_*` kernel tuning
environment variables are retained for the validated Sinkhorn benchmark scripts.

## Full eight-panel paper benchmark

Install all benchmark and plotting dependencies in the isolated environment:

```bash
python -m pip install -e '.[dev,plots,baselines]'
bash scripts/setup_ott_hessian.sh
python -c "import torch, pykeops, jax; print(torch.__version__); print(pykeops.__version__); print(jax.devices())"
```

The baseline extra pins the paper-compatible package versions and uses JAX's
`cuda12-local` plugin so the server's CUDA 12.8/cuDNN installation is reused;
it does not replace PyTorch's pinned NVIDIA CUDA wheels.

After one GPU has been allocated, run the complete synthetic benchmark:

```bash
bash scripts/run_paper_benchmarks.sh 1
```

The defaults match Appendix H of the paper: points sampled uniformly from
`[0,1]^d`, uniform marginals, full squared-Euclidean cost, `epsilon=0.1`, ten
fixed forward/backward iterations, and 100 Sinkhorn plus 50 fixed CG iterations
for HVP. It uses 10 warmups and 50/30/20 measured forward/backward/HVP runs,
with TF32 for forward/backward and strict FP32 for HVP. Sizes run from large to
small. The symmetric schedule updates both potentials in one fused Triton
launch. On a 16 GB RTX 5080,
known-quadratic Tensorized cases are skipped before OOM according to
`--max-tensorized-mib`; the CSV records every skip or baseline failure.

A short end-to-end check before the long run is:

```bash
bash scripts/run_paper_benchmarks.sh 1 \
  --n-sizes 5000 --d-sizes 64 \
  --hvp-n-sizes 5000 --hvp-d-sizes 64 \
  --warmups 1 --forward-repeats 2 --backward-repeats 2 --hvp-repeats 2
```

Each run writes `00_overview.png`, eight numbered per-panel PNGs,
`paper_results.csv`, `paper_results.json`, and `environment.json` under a new
`outputs/paper_<UTC timestamp>/` directory. Results are checkpointed after
each method/size. KeOps and JAX use streaming online backends; Tensorized is
omitted from HVP as in the paper, and KeOps/JAX are omitted from PyTorch peak
memory plots because their external allocators would make those values invalid.
Absolute runtimes will not match the paper's A100-80GB; compare curve shape,
OOM boundary, and speedup ratios instead. Bounded GPU autotuning is available
with `--autotune --block-m 64 --block-n 128`. Candidates exceeding the smaller
of the device's shared-memory limit and 64 KiB are rejected before launch.
Compilation and tuning run in warmup; `autotuning.json` records every candidate
and the selected configuration. The default retains fixed RTX-safe tiles.

The HVP defaults extend FlashSinkhorn and KeOps to `n=50,000`, while JAX
stops at `n=10,000`; all three use at most `d=128` except FlashSinkhorn,
which extends to `d=512`. JAX HVP now calls the author's external OTT-Hessian
HessianA/OTT geometry implementation from a pinned checkout. A guarded fixed-step
CG adapter matches Flash/KeOps; `tau2=damping/epsilon` preserves the same
absolute Schur damping as Flash and KeOps. Potentials are shared dynamic JIT
inputs, and solve/setup stays outside HVP timing. The source hash and numerical
controls are recorded. This is an equivalent baseline with a configured CG solve,
not proof that this upstream revision generated the paper tables.
The earlier custom implementation remains available with
`--jax-hvp-backend matrix-free` and is labelled separately. Missing upstream
source is reported as a failed baseline, never silently replaced.
Vectors now use a dedicated transport reduction. Source/target gradients
reuse a score tile across all feature blocks and retain actual marginal masses
even for unconverged solves. The Tensorized cost matrix is deliberately precomputed
outside forward timing, as in the official benchmark. Consequently, a
Tensorized win at `d=1024` is expected rather than a correctness failure;
Appendix H, Tables 10--11 report the same crossover on A100.
Actual CUDA compile success and speedups must be checked on the server after
pulling this change.
Memory panels contain only alternating FlashSinkhorn and Tensorized, and HVP
panels contain symmetric FlashSinkhorn, KeOps and JAX.

Before a full run, validate and compare the retained generic kernels with the
specialized and tuned implementations on exactly the same tensors:

```bash
bash scripts/validate_paper_optimizations.sh 1 --cases 10000:64 20000:1024 --profile
bash scripts/run_paper_benchmarks.sh 1 --autotune --block-m 64 --block-n 128 --diagnostics
```

The first command requires real GPU tests to pass, writes complete HVP and
forward+backward samples, numerical comparisons, residual/CG diagnostics and
separate profiler traces. The second writes all eight panels with diagnostics
outside timing. See [reproduction checks](docs/paper_reproduction.md) for setup,
protocol differences and how to transfer unpushed changes to the server.

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
from flashsinkhorn import (sinkhorn_flash, sinkhorn_cost, apply_plan,
                           diagnostics, point_gradients,
                           hessian_vector_product)

# Khởi động Python với CUDA_VISIBLE_DEVICES đã trỏ tới GPU được cấp.
x = torch.randn(256, 32, device="cuda", dtype=torch.float32) * 0.1
y = torch.randn(384, 32, device="cuda", dtype=torch.float32) * 0.1
result = sinkhorn_flash(x, y, epsilon=0.2, n_iters=300, tol=1e-4)
print(diagnostics(result))
py = apply_plan(result, y)  # P @ y, không tạo P
grad_x, grad_y = point_gradients(result)  # envelope gradient khi đã hội tụ

# Scalar loss với analytic backward; không backprop qua các vòng Sinkhorn.
x = x.requires_grad_(True)
loss = sinkhorn_cost(
    x, y, epsilon=0.2, n_iters=300, tol=1e-4,
    backend="flash", precision="tf32",
)
grad_x = torch.autograd.grad(loss, x, create_graph=True)[0]
direction = torch.randn_like(x)
hvp_x = torch.autograd.grad((grad_x * direction).sum(), x)[0]

# Low-level API khi đã có potentials và cần thông tin hội tụ CG.
hvp_x, cg_info = hessian_vector_product(
    result, direction, damping=1e-5, max_cg_iters=50,
    cg_rtol=1e-6, return_info=True,
)
```

Backward dùng gradient analytic từ barycentric projection. HVP theo Theorem 5
của paper: Schur complement có damping + CG, các phép `P@v`, `P.T@v`, `P@M`
và Hadamard-weighted transport đều streaming, không materialize `P`. HVP hiện
chỉ hỗ trợ double backward theo `x` khi giữ `y` cố định; các transport nội bộ
của HVP dùng IEEE FP32 như khuyến nghị số của paper. Potentials phải hội tụ
trước khi gradient/HVP được xem là đạo hàm của nghiệm EOT tối ưu.

`materialize_plan` chỉ dành cho kiểm tra bài toán nhỏ và có giới hạn kích thước.
Không hỗ trợ unbalanced OT trong bản này dù tên thư mục cha có “FlashUOT”.

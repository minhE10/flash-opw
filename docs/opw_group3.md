# Nhóm 3: tuning công bằng trên TRAIN

Runner: `python -m experiments.opw_group3`. Đây là tìm tham số cho các
implementation trong repo, không chạy code tác giả và không đánh giá TEST.
FlashOPW vẫn dùng cost affine và score Eq.19 của `main (2).pdf`.

## Protocol khai báo trước khi chạy

* Ba seed `20261006 20261007 20261008`. Mỗi seed tách stratified holdout
  trong official TRAIN bằng cùng hàm `training_holdout` của nhóm 1–2. Gallery
  và query không giao nhau trong một split; mỗi class có mẫu ở cả hai phía.
  Các split có thể dùng lại mẫu, nên không phải ba tập dữ liệu độc lập.
* Pilot mặc định: 16 gallery, 14 query mỗi split; `--gallery 28 --queries 28`
  mở rộng trên server. `0` dùng toàn bộ phần TRAIN được phân chia tương ứng.
  Không đọc nhãn TEST, không chọn split hoặc candidate theo TEST.
* Giữ preprocessing của archive, squared-Euclidean ground cost, cost scale=1,
  uniform marginals. Làm tròn input về FP32 rồi dùng cùng giá trị cho FP64/FP32.
  Việc này không khẳng định protocol feature/ground cost khớp mọi bảng journal.
* Chọn một candidate cho mỗi metric bằng **mean ACC@1 trên ba split**, sau đó
  **mean MAP trên ba split**, cuối cùng candidate xuất hiện sớm hơn. Không chọn
  ba model riêng theo từng seed. SD là sample SD giữa split, không phải CI.
* MAP tính trên toàn gallery của split; vote/tie rules của `retrieval.py`.
  Không tuning k. Sau selection, đánh giá TEST độc lập thuộc nhóm 4.
* Cùng tiêu chí OT regularized: `max(row_L1,col_L1) <= 1e-3`, kiểm tra mỗi
  50 vòng, cap 4000, schedule f rồi g. Dừng từng cặp ở checkpoint đầu đạt.
  Ngưỡng marginal không phải bảo đảm sai số objective phổ quát.
* Lưu toàn bộ cặp chưa đạt cap và score của chúng. Candidate có bất kỳ cặp
  chưa đạt trên bất kỳ split nào không đủ điều kiện để freeze. Không bỏ riêng
  các cặp khó rồi tính accuracy trên tập nhỏ hơn. Nếu một metric không còn
  candidate hợp lệ, run trả lỗi và selection mang trạng thái `failed`.
* DTW, LDTW, NDTW và unregularized OT không search regularization; mỗi metric
  có một candidate. LP phải báo solver success. Marginal/iterations trong
  các matrix DTW/Soft-DTW/LP là số 0 làm placeholder, không phải residual đo
  được hoặc bằng chứng hội tụ Sinkhorn.

## Candidate và ngân sách

Mặc định **6 candidate cho từng metric có tham số**, đã bao gồm preset.
Tổng 46 candidate × 3 split = **138 job**. Ngân sách bằng số candidate thử,
không bằng thời gian wall. Candidate thất bại hội tụ vẫn tính vào ngân sách.
`--budget 2..12` chọn prefix cố định; đây là joint search nhỏ đã khai báo,
không phải full Cartesian grid hoặc tuyên bố tìm được optimum.

FacesUCR, sáu candidate mặc định:

| Metric | Các giá trị được thử theo thứ tự |
|---|---|
| FlashOPW | `(mu,epsilon)`: `(1.05,.1),(50,.1),(10,.3),(50,.3),(10,.03),(200,.1)` |
| TLp | `(weight,epsilon)`: `(50,.1),(1.05,.1),(10,.3),(50,.3),(10,.03),(200,.1)` |
| OPW journal | `(lambda1,lambda2,sigma)`: `(1,.1,1),(1,.1,3),(.1,.3,1),(10,.1,3),(50,.03,5),(1,.3,5)` |
| OPW-KL journal | `(lambda2,sigma)`: `(.1,1),(.1,3),(.3,1),(.03,5),(.3,5),(.1,5)` |
| Sinkhorn | epsilon `.1,.03,.3,.01,1,.003` |
| TCOT | lambda `1,10,.1,3,.3,30`, entropy epsilon=`1/lambda` |
| Soft-DTW | gamma `.1,.03,.3,.01,1,.003` |

Flash dùng `mu=lambda1+lambda2/(2*sigma^2)` và tính sigma theo mu; không search
thêm các tổ hợp lambda1/sigma dư thừa có cùng coupling. Với mu nhỏ hơn lambda1
preset, dùng lambda1=mu/2. Candidate 0 giữ literal preset sigma=1.
**Sigma journal giữ đơn vị perpendicular-distance của journal**, không copy
sigma từ Flash. OPW-KL không có inverse-order penalty (lambda1=0).

FaceAll dùng lambda1 preset=10; dataset khác dùng50. Flash preset mu thay đổi
tương ứng, và TLp có cùng tập mu/epsilon với Flash. Các candidate được ghi
đầy đủ vào `candidate_manifest.json` **trước khi giải**; file chứa cả indices
ba split và stopping policy. File freeze cũng giữ toàn bộ grids.

Các score riêng giữ nguyên: Flash Eq.19; TLp `<P,D+wF>`; TCOT
`<P,D*(1+abs(delta))>`; Sinkhorn/OPW/OPW-KL `<P,D>`; Soft-DTW raw accumulated
smoothed cost (không chuyển sang divergence). DTW không dùng window.

## Chạy trên server

Tiếp tục dùng GPU1 đã được cấp và environment `minh`:

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
bash scripts/validate_flash_opw.sh 1
bash scripts/run_opw_group3.sh 1 \
  --budget 6 --gallery 28 --queries 28 \
  --output outputs/opw_group3_gpu_20261006
python -m experiments.opw_group3_audit outputs/opw_group3_gpu_20261006
```

Đây là 784 cặp mỗi split/candidate; tổng108192 cặp, gồm CPU DTW/Soft-DTW/LP
và CUDA entropic distances. Đây là tuning chất lượng, **không so speedup**.
Flash là streaming IEEE FP32; entropic baselines là dense Torch FP32 trên cùng
GPU; CPU local là explicit FP64 oracle. Timings search gồm conversion,
diagnostics, evaluation và compilation, không phải kernel latency.

Job hoàn tất được cache thành NPZ và JSON; resume bằng **nguyên lệnh trên**
thêm `--resume`. Resume kiểm tra source hashes, data, grids, split indices,
settings, Python/Torch/packages và GPU. Không nâng budget/cap trong cùng output.
Muốn một protocol khác, dùng output mới và báo riêng.

Sau run thành công, gửi **nguyên thư mục** như nhóm2 hoặc ZIP chứa toàn bộ:

```bash
tar -czf outputs/opw_group3_gpu_20261006.tar.gz \
  -C outputs opw_group3_gpu_20261006
```

## Output và freeze

* `preset_results.csv` / bảng preset trong `comparison_report.md`: tham số
  preset journal của repo, được giải đến cùng ngưỡng marginal. Không gọi đây
  là runtime/quality ở số vòng20/100 của journal.
* `tuned_results.csv` / bảng tuned: tất cả metric search trên cùng ba split.
  Bảng này là validation sau search, có selection bias, không phải TEST.
* `candidate_results.csv`, `evaluations.json`: mean/SD, từng split, cap counts,
  số vòng, search time, AP và predictions; các candidate không hợp lệ vẫn lưu.
* `matrices/`: distance, residual, iterations, labels và TRAIN indices của
  từng job. `jobs/`: kết quả và SHA256 của NPZ, dùng resume/audit.
* `environment.json`, `run_state.json`, `candidate_manifest.json`: provenance,
  trạng thái và protocol. `selected_all_metrics.json`: tham số đã freeze.

Artifact mới có schema `flashopw-all-metrics-selection-v1`, gồm tất cả metric,
stopping policy và score conventions. Consumer nhóm4 dùng
`experiments.opw_group3.load_frozen(path,dataset=...,training_sha256=...)`;
loader từ chối artifact thiếu metric, dùng TEST, incomplete hoặc có cap failure.
File này **không** thay thế hoặc ghi đè JSON Flash-only cũ, và không truyền nó
cho `opw_knn --flash-parameters` (runner cũ dùng fixed iterations/schema khác).
Nhóm4 sẽ dùng policy và tham số mới cho cả presets lẫn tuned trước khi so TEST.

Local reference run dùng `.venv-baselines/Scripts/python.exe -m
experiments.opw_group3 --device cpu --budget 6 --output
outputs/opw_group3_cpu_20261006`. Kết quả CPU và GPU phải giữ thư mục riêng.

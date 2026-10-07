# FlashOPW theo main.pdf so với OPW gốc của journal

Theo yêu cầu ngày 07/10/2026, hai đối tượng chính là **FlashOPW trong
`main (2).pdf`** và **OPW gốc trong `OWD_journal.pdf`**. `affine-opw-dense`
chỉ là backend dense của công thức main; không được gọi là OPW gốc.
`opw-exact-relative` của nhóm 2 là đối chứng riêng, không thay OPW journal.

## Công thức được giữ nguyên

Đặt `R=(i/N-j/M)^2`, `h=1/N^2+1/M^2`,
`D=cost_scale*||x_i-y_j||^2`, epsilon=`lambda2`, trọng số uniform.

| Thành phần | FlashOPW: main | OPW gốc: journal |
|---|---|---|
| Inverse temporal penalty | Taylor: `1-R` | Exact: `1/(1+R)` (Eq.10,16) |
| Gaussian squared distance | `R` | `R/h` (Eq.11–12) |
| Cost đầy đủ | `D+(lambda1+lambda2/(2*sigma^2))*R+q0` | `D-lambda1/(1+R)+lambda2*(R/(2*sigma^2*h)+log(sigma*sqrt(2*pi)))` |
| Score dùng để xếp hạng | Literal Eq.19: `a.f+b.g-lambda2+q0` | Eq.13: `<P,D>` |
| Backend trên GPU | Flash, IEEE FP32, cost qua augmented features | Dense FP32, inverse exact, log-Sinkhorn |

`q0=lambda2*log(sigma*sqrt(2*pi))-lambda1`. Trừ hằng số này khỏi cost
journal cho ta `D+lambda1*R/(1+R)+lambda2*R/(2*sigma^2*h)`.
Đây là phép biến đổi đại số chính xác để tính coupling ổn định, không phải Taylor.
Score journal vẫn chỉ là `<P,D>`, không phải cost regularized hay Eq.19 của main.

Không đổi sigma journal sang đơn vị relative của main. Mỗi phương pháp giữ tham số
đã freeze riêng sau cùng budget tuning TRAIN. Với FacesUCR hiện tại:

| Profile | Phương pháp | lambda1 | lambda2 | sigma |
|---|---|---:|---:|---:|
| preset | FlashOPW | 1 | 0.1 | 1 |
| preset | OPW journal | 1 | 0.1 | 1 |
| selected | FlashOPW | 1 | 0.1 | 0.031943828249996996 |
| selected | OPW journal | 1 | 0.3 | 5 |

Hai sigma cùng giá trị số không có cùng độ rộng prior vì khác đơn vị.
Hai cost, epsilon và score cũng có thể khác nhau; do đó lý thuyết không buộc
MAP/ACC hoặc nearest neighbor bằng nhau. So sánh này đánh giá hai phương pháp
đúng định nghĩa, không kết luận riêng tác động của Taylor. Nhóm 2 giữ vai trò
ablation các yếu tố; nhóm 1 giữ vai trò kiểm tra Flash so với dense của cùng cost.

## Kiểm tra và kết quả đã có

Runner `experiments.opw_main_vs_journal` chỉ đọc TRAIN và sử dụng cùng ba split
đã freeze: 28 query × 28 gallery mỗi split. Điểm dừng chung là cả hai marginal
L1 <= 1e-3, kiểm tra mỗi 50 vòng, cap 4000. Schedule chung `f_then_g` là lựa chọn
của implementation hiện tại; Algorithm 2 journal trình bày `g_then_f`. Đối chứng
số vòng cố định dùng schedule chung này, không tuyên bố tái hiện từng bước của
Algorithm 2 hay đang chạy code tác giả. Spatial ground cost chung là squared L2
theo main, không tự động tái hiện mọi thiết lập dữ liệu/ground metric của journal.

Runner kiểm tra fingerprint TRAIN, source hash, tham số, indices/labels,
SHA256 của ma trận; tính lại MAP và ACC@1. Với mỗi ma trận, một cặp được giải
lại bằng SciPy FP64 theo đúng số vòng GPU đã lưu, kiểm tra cost từ công thức
literal của PDF và score tương ứng. Đây là kiểm tra score có lấy mẫu, không phải
kiểm tra toàn bộ coupling của mọi cặp. Kết quả GPU cũ được tái sử dụng có ghi rõ
provenance; việc đọc lại trên Windows không được ghi là chạy GPU mới.

| Profile | Phương pháp | ACC@1 trung bình | MAP trung bình | Cặp chưa đạt tolerance |
|---|---|---:|---:|---:|
| preset | FlashOPW | 58.333% | 58.464% | 0 |
| preset | OPW journal | 77.381% | 72.899% | 1 |
| selected | FlashOPW | 79.762% | 74.405% | 0 |
| selected | OPW journal | 76.190% | 71.397% | 0 |

Sau tuning, chênh lệch là +3.571 điểm phần trăm ACC@1 và +3.008 điểm MAP cho
FlashOPW. Đây là pilot TRAIN sau selection, chưa chứng minh tổng quát hóa hoặc
ưu thế trên TEST. Preset journal được giữ trong báo cáo nhưng không đạt điều kiện
hội tụ toàn bộ. Không dùng tỉ số của hai raw distance để đánh giá equivalence
vì score có định nghĩa khác nhau. Thời gian tuning/cached audit không phải benchmark tốc độ.

Báo cáo kiểm tra: [comparison_report.md](../reports/opw_main_vs_journal_20261007/comparison_report.md),
[summary.json](../reports/opw_main_vs_journal_20261007/summary.json).

## Lệnh trên server

Đọc lại kết quả đã chạy, không cần tuning hoặc solve GPU lại:

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
python -m experiments.opw_main_vs_journal \
  --reuse-group3 outputs/opw_group3_gpu_20261006 \
  --output outputs/opw_main_vs_journal_verified_20261007
```

`--reuse-group3` yêu cầu folder gốc đầy đủ `matrices/`, `jobs/`, manifest và
selection. Folder output mới phải chưa tồn tại; thêm `--resume` nếu tiếp tục
cùng cấu hình. Runner audit lại đủ 138 job nhóm 3 trước khi dùng 12 ma trận
hai phương pháp/preset/selected cho báo cáo riêng.

Nếu cần chạy kiểm tra GPU mới cho hai phương pháp đã tuning, thêm phép so
200 vòng cố định bên cạnh common tolerance:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m pytest tests/test_opw_main_vs_journal.py --require-gpu -q

bash scripts/run_opw_main_vs_journal.sh 1 \
  --profiles selected --fixed-iters 200 \
  --output outputs/opw_main_vs_journal_fresh_20261007
```

GPU `1` là GPU đã được cấp như các lần chạy trước; wrapper giữ allocation và
giới hạn memory fraction 0.45, hai CPU thread. Không sweep hyperparameter mới.
File output gồm `split_results.csv`, `mean_results.csv`, `summary.json`,
`comparison_report.md` và ma trận/job của fresh solve. Ngưỡng hội tụ vẫn được
báo ở chế độ 200 vòng; không coi mọi cặp 200 vòng là đã hội tụ. Không yêu cầu
hai phương pháp có cùng coupling/score; chỉ kiểm tra mỗi phương pháp với
FP64 oracle theo công thức của chính nó.

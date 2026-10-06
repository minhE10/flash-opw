# Pilot tuning FlashOPW — 06/10/2026

Đã thực hiện chọn tham số bằng TRAIN rồi đánh giá một subset TEST riêng trên
CPU FP64. Giữ nguyên cost affine, thời gian tương đối và Eq.19 của main PDF.
Kết quả có triển vọng; chưa xác nhận accuracy hay tốc độ của preset mới trên GPU.

## Protocol và kết quả

FacesUCR official TRAIN200/TEST2050, 14 class, 131 frame/d1; giá trị gốc không
đổi. Tuner seed20261006 chia holdout trong TRAIN; balanced gallery28 và
validation14, chỉ số hai phần không giao nhau. Thử 3 ứng viên:
`mu={1.05,50,430}`, epsilon0.1, cost_scale1, 200 vòng f rồi g, pdf-loss.
ACC@1 ưu tiên, MAP phá hòa. Không đọc TEST khi chọn tham số.

| Ứng viên | ACC@1 validation | MAP validation |
|---|---:|---:|
| mặc định, mu1.05 | 57.143% | 60.815% |
| mu50 — được chọn | 85.714% | 74.278% |
| mu430 | 64.286% | 65.933% |

File [selected_parameters.json](selected_parameters.json) được đóng băng
trước phép đánh giá TEST. Tham số chọn là `lambda1=1`, `lambda2=.1`,
`sigma=.031943828249997`, tức `mu=50`. Đây là lựa chọn từ grid 3 ứng viên
nhỏ, không phải kết quả grid mặc định 18 ứng viên hay một giá trị tối ưu tổng quát.

Đánh giá TEST gồm gallery32 lấy từ toàn bộ TRAIN bằng seed20261007;
query28 lấy từ TEST bằng seed20261008. Hai phương pháp dùng cùng gallery,
query, 200 vòng, score và FP64 oracle. MAP tính trên toàn gallery32.

| Phương pháp | ACC@1 | ACC@3 | ACC@5 | ACC@7 | ACC@15 | ACC@30 | MAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| mặc định | 53.571% | 53.571% | 53.571% | 42.857% | 17.857% | 25.000% | 59.854% |
| tham số đã chọn | 92.857% | 89.286% | 78.571% | 64.286% | 28.571% | 25.000% | 81.986% |

ACC@1 tăng **39.286 điểm phần trăm**, MAP tăng **22.132 điểm phần trăm**.
Không so trực tiếp các số này với pilot server 16 query trước đó vì subset
khác. Tập TEST đã từng được dùng cho phân tích pilot/ablation; quy trình mới
không lấy nhãn test để chọn ứng viên, nhưng chưa phải một đánh giá hoàn toàn
blind của quá trình phát triển phương pháp. Chỉ có một split validation/test
nhỏ nên chưa kết luận hiệu quả trên toàn FacesUCR hoặc các dataset khác.

Sai số marginal tối đa trên 896 cặp test tăng từ `6.432e-4` ở mặc định lên
`1.185e-2` ở bản chọn; median từ `6.731e-6` lên `2.458e-3`. Preset mạnh hơn
**chưa hội tụ hoàn toàn sau 200 vòng**. Kết quả là accuracy của thuật toán ở
số vòng cố định, không chứng minh chất lượng của nghiệm OT đã hội tụ. Tốc độ
GPU và parity với CPU phải được đo trên server với bộ tham số này.

## Kiểm tra và dữ liệu lưu

[validation.csv](validation.csv), [search.json](search.json),
[test_comparison.json](test_comparison.json) lưu phép chọn, source hashes,
dataset hashes, chỉ số và AP/predictions. Hai file `test_default.npz`,
`test_selected.npz` chứa score matrices, residuals và labels để recompute MAP/ACC.
Metadata [environment.json](environment.json) là CPU local, không phải RTX5080.
Các source hashes ghi snapshot lúc chạy trước một số thay đổi runner/benchmark;
engine FlashSinkhorn/FlashOPW không thay đổi trong lần cải tiến này.

Để tạo lại lựa chọn nhỏ bằng code hiện tại:

```bash
python -m experiments.opw_tune --device cpu --dataset FacesUCR \
  --gallery 28 --validation 14 --mus 1.05 50 430 --epsilons .1 \
  --output outputs/opw_cpu_tune_recheck
python reports/opw_tuning_20261006/evaluate_cpu_pilot.py
```

Script đánh giá đọc file selection đã đóng băng trong report, không search TEST.
Để xác nhận GPU, có thể dùng chính file selection này với `--flash-parameters`,
hoặc chạy grid 18 ứng viên bằng tuner GPU rồi đóng băng một output mới.
Xem [quy trình server và chuỗi dài](../../docs/opw_tuning_scaling.md).

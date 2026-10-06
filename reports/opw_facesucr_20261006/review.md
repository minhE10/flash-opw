# Kiểm tra pilot FlashOPW trên server — 06/10/2026

Đã đọc `results.tar.gz` từ commit `9fe0a27`. Kết luận: **FlashOPW CUDA
tính đúng kết quả affine theo main ở preset 200 vòng trên pilot này**.
Chưa có bằng chứng phương pháp sửa đổi cải thiện chất lượng phân loại so
với OPW journal hoặc tái hiện các bảng journal.

Đã chạy thêm đối chứng để tách Taylor, prior normalization, score và số
vòng: xem [báo cáo Taylor](taylor_ablation.md). Đây là phân tích trên cùng
pilot, không thay công thức production theo main.

## Dữ liệu, môi trường và tính nhất quán

- FacesUCR: 32 gallery/16 query, 512 cặp mỗi metric; 12 metric hoàn tất,
  tổng 6144 score hữu hạn, không có failure.
- RTX 5080, GPU vật lý 1; Torch 2.11.0+cu128, Triton 3.6.0, IEEE FP32.
  Dense references dùng CPU FP64.
- `lambda1=1`, `lambda2=0.1`, `sigma=1`, `cost_scale=1`;
  FlashOPW/affine dense 200 vòng, journal OPW/OPW-KL 20 vòng.
- Score FlashOPW đúng literal main (19), không phải spatial `<P,D>`.
- Cả 9 source hashes trong resume signature khớp checkout `9fe0a27` lúc review.
  `git_dirty=true` trong metadata không phủ nhận kiểm tra hash này;
  không suy diễn rằng toàn bộ worktree trên server là sạch.
- SHA256 TRAIN/TEST và các chỉ số subset khớp dữ liệu local; labels trong
  toàn bộ matrices khớp split được chọn. Mọi query có gallery cùng class.
- Đã tính lại MAP, ACC, AP và predictions từ tất cả matrices: khớp chính
  xác JSON và cả 72 dòng CSV, gồm thời gian trong checkpoint.

## Đối chứng số học

| Kiểm tra trên đủ 512 cặp | Kết quả |
| --- | ---: |
| Flash FP32 vs dense FP64, max absolute score error | `9.58348e-8` |
| Relative L2 score error | `5.02733e-8` |
| Ranking toàn gallery của cả 16 query | Giống hoàn toàn |
| Predictions ở cả sáu k và AP từng query | Giống hoàn toàn |
| Khoảng cách score nhỏ nhất giữa hạng 1 và hạng 2 trong dense | `2.68714e-4` |
| Server dense vs local dense 200 vòng, max absolute error | `1.11022e-16` |

Sai số FP32 không làm thay đổi kết quả retrieval/classification của
đối chứng affine trong pilot. Đây là kiểm chứng trên FacesUCR d=1, không
thay thế bộ GPU tests nhiều chiều và nhiều schedule. Archive không chứa
log `validate_flash_opw.sh`, nên chưa xác nhận bộ test đó đã pass.

## MAP và ACC (%)

MAP dùng toàn bộ 32 gallery đã chọn; không phải full-split MAP journal.

| Metric | MAP | ACC@1 | ACC@3 | ACC@5 | ACC@7 | ACC@15 | ACC@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FlashOPW | 68.449 | 62.50 | 62.50 | 68.75 | 62.50 | 31.25 | 37.50 |
| Affine dense | 68.449 | 62.50 | 62.50 | 68.75 | 62.50 | 31.25 | 37.50 |
| DTW | 62.778 | 62.50 | 56.25 | 56.25 | 62.50 | 43.75 | 37.50 |
| lDTW | 62.778 | 62.50 | 56.25 | 56.25 | 62.50 | 43.75 | 37.50 |
| nDTW | 52.337 | 37.50 | 56.25 | 50.00 | 50.00 | 37.50 | 37.50 |
| Soft-DTW | 69.055 | 75.00 | 68.75 | 62.50 | 62.50 | 37.50 | 37.50 |
| OT | 56.937 | 56.25 | 56.25 | 62.50 | 68.75 | 43.75 | 31.25 |
| Sinkhorn | 57.363 | 56.25 | 56.25 | 62.50 | 75.00 | 43.75 | 31.25 |
| TLp | 82.482 | 81.25 | 81.25 | 81.25 | 75.00 | 43.75 | 31.25 |
| OPW-KL | 84.034 | 93.75 | 93.75 | 87.50 | 75.00 | 56.25 | 31.25 |
| TCOT | 64.225 | 62.50 | 68.75 | 68.75 | 62.50 | 43.75 | 31.25 |
| OPW journal | 84.034 | 93.75 | 93.75 | 87.50 | 75.00 | 56.25 | 31.25 |

OPW và OPW-KL có cùng MAP/ACC, nhưng matrices khác nhau tới `0.00180778`
và full rankings cũng khác; việc bằng nhau ở chỉ số tổng hợp không phải
bằng chứng hai implementation bị gọi nhầm hoặc cho cùng coupling.
16 query khiến mỗi dự đoán đúng/sai thay đổi ACC 6.25 điểm phần trăm;
chưa chọn tham số/k hoặc kết luận tổng quát từ subset này.

## Hội tụ

Diagnostics server chỉ chứa 8 cặp của query đầu: row L1 từ `6.44e-6`
đến `8.68e-4`; column L1 từ `4.81e-7` đến `6.16e-7`; mass gần 1.

Đã dựng lại **cả 512 coupling bằng recurrence FP64 độc lập** từ cost
tường minh, 200 vòng, f rồi g. Score khớp archive tới `2.22e-16`.

| Marginal L1 trong audit FP64 | Kết quả |
| --- | ---: |
| Median row L1 | `4.55590e-6` |
| Maximum row L1 | `8.67810e-4` |
| Maximum column L1 | `1.20997e-15` |
| Số cặp vượt `1e-3` | 0 / 512 |
| Số cặp vượt `1e-4` | 57 / 512 |
| Số cặp vượt `1e-5` | 198 / 512 |

Vậy 200 vòng giải đúng preset, gần khả thi ở ngưỡng `1e-3`, nhưng chưa
thể coi mọi cặp hội tụ ở `1e-5`. Audit toàn bộ là FP64, không phải đo lại
mọi marginal trên GPU; archive chỉ cung cấp 8 marginal GPU.

Chạy thêm dense trên 8 cặp diagnostics với cap 1000, tol `1e-8`:
6 cặp đạt tol, 2 cặp hết cap và vẫn còn row residual tới `2.64e-6`.
Score thay đổi tối đa `2.13300e-5` so với 200 vòng. Chưa kiểm tra
ranking khi tăng số vòng trên toàn bộ 512 cặp. Gap âm trong diagnostics
không chứng nhận optimality khi coupling chưa có đúng marginal.

## Vì sao cần tách việc đúng kernel và việc cải thiện metric

Với `N=M=131`, đặt `Delta=i/N-j/M`. Main có hệ số temporal tổng
`mu=1+0.1/(2*1²)=1.05` trước `Delta²`. Riêng Gaussian term của journal có
hệ số `0.1/[2*1²*(1/N²+1/M²)]=429.025` trước `Delta²`, cộng inverse term
riêng. Cùng `sigma=1` không mang cùng độ mạnh ràng buộc thời gian giữa
hai định nghĩa prior.

Đây là khác biệt đại số đã xác minh, chưa phải kết luận nhân quả rằng nó
giải thích toàn bộ khoảng giảm accuracy. Còn ảnh hưởng Taylor, score
main (19) vs `<P,D>`, số vòng và tham số chưa được chọn cho định nghĩa
main. Sai số GPU vs affine dense không giải thích khoảng giảm MAP/ACC.

## Thời gian và hướng tiếp theo

512 cặp: FlashOPW 13.1206 s, affine dense CPU 101.7103 s; tỷ số wall time
7.75. Trung bình FlashOPW 25.63 ms/cặp. Backend/precision khác nhau và
chưa có repeated timing/peak-memory measurements; không coi tỷ số này
là speedup riêng của thuật toán hoặc của kernel.

Metadata đầu job: `15271/16303 MiB` VRAM đã dùng; Torch báo còn khoảng
580 MiB trống. Điều đó chưa chứng minh GPU compute bị tranh chấp, nhưng
cần ghi lại điều kiện tải khi benchmark tốc độ. Không suy diễn hoặc
thay đổi allocation của job khác.

Ưu tiên tiếp theo:

1. Kiểm tra số vòng/tolerance trên cùng các cặp, lưu marginal toàn bộ và
   xác nhận ranking ổn định. Không dùng 200 vòng làm bằng chứng optimum.
2. Ablation giữ nguyên main cost nhưng dùng spatial score; đồng thời so
   với inverse chính xác dùng cùng prior relative và cùng số vòng để
   tách ảnh hưởng Taylor khỏi score và normalization.
3. Chọn `sigma/lambda1/lambda2` bằng validation trong training split, giữ
   test split để đánh giá. Không ép số của journal bằng đổi công thức main.
4. Profile workload 131 frame/d=1; xem batching nhiều cặp và giảm số lần
   launch/host synchronization. Đây là hướng tối ưu cần đo, chưa xác
   nhận bottleneck bằng profiler. Đối chứng tốc độ cần cùng GPU/precision.

Một run ablation nhỏ có thể dùng runner hiện tại, trên GPU 1 đã được cấp:

```bash
bash scripts/run_opw_knn.sh 1 --datasets FacesUCR \
  --max-train 32 --max-queries 16 \
  --metrics flash-opw affine-opw-dense opw-exact-relative \
  --opw-score spatial --iters 200 --journal-opw-iters 200 \
  --output outputs/opw_facesucr_spatial_ablation_20261006
```

Các kiểm tra local được lưu riêng trong
`outputs/opw_server_review_20261006/audit.json`,
`full_convergence_summary.json`, `full_dense_200_convergence.npz`,
`convergence_checks.json`. Archive gốc được đọc tại chỗ, không thay đổi.

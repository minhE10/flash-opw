# Context bàn giao FlashOPW / FlashSinkhorn

Cập nhật ngày **07/10/2026**, múi giờ Asia/Jakarta. File này ghi lại trạng thái tại thời điểm bàn giao; tiến độ server có thể đã tăng sau đó.

Để tiếp tục ở chat mới, hãy đọc file này và `docs/opw_experiment_plan.md`, sau đó kiểm tra code và Git hiện tại trước khi thay đổi.

## 1. Workspace, server và cách làm việc

- Repo Windows: `F:\HUST\Machine Learning\Optimal Transport\FlashUOT\flash-opw`.
- Remote: <https://github.com/minhE10/flash-opw>, branch `main`.
- Commit code gần nhất đã push tại thời điểm bàn giao: **`3813c08` — Fix FP32 checkpoint validation and preserve legacy group4 resume**.
- Server: `/home/doanpt/minh.nd/flash-opw`; conda environment `minh`; GPU RTX 5080.
- Người dùng tự chạy lệnh trên server rồi gửi kết quả; chưa thiết lập SSH cho agent.
- GPU vật lý **1**, ánh xạ thành `cuda:0` qua `CUDA_VISIBLE_DEVICES=1`. Giữ CPU threads=2 và memory fraction=0.45 theo protocol hiện tại.
- Windows local chỉ có môi trường `.venv-baselines` với PyTorch CPU; không được báo đã kiểm tra CUDA locally.
- Người dùng đã cho phép commit/push thay đổi phục vụ công việc này. Việc lưu file bàn giao không tự đồng nghĩa cần push ngay.
- Deadline: **báo cáo chiều 08/10/2026**.
- Archive kết quả lớn từng được gửi lên GitHub. Sau khi audit, giữ bản gốc trong `outputs/`, báo cáo nhỏ trong `reports/`, rồi xóa archive khỏi cây file được theo dõi. Xóa bằng commit thường vẫn giữ archive trong lịch sử Git.

## 2. Mục tiêu và định nghĩa phương pháp

Ban đầu người dùng tái hiện FlashSinkhorn bằng code tự triển khai. Đã thử chuyển sang code tác giả, sau đó yêu cầu revert; hiện tiếp tục dùng bản tự reproduce. FlashSinkhorn và FlashOPW được tách thành package `flashsinkhorn` và `flashopw`.

Tài liệu nguồn nằm tại thư mục cha repo:

- `main (2).pdf`: **ưu tiên cho FlashOPW sửa đổi**.
- `OWD_journal.pdf`: định nghĩa **OPW gốc** và các metric đối chiếu.

So sánh chính là **FlashOPW theo main với OPW gốc theo journal**. Không thay OPW journal bằng dense affine OPW.

Với `R=(i/N-j/M)^2`, `h=1/N^2+1/M^2` và `D` là squared L2 có cost scale:

| Phương pháp | Chi phí |
|---|---|
| FlashOPW main | `C = D + mu*R + q0`, `mu=lambda1+lambda2/(2*sigma^2)`, `q0=lambda2*log(sigma*sqrt(2*pi))-lambda1` |
| OPW journal | `C = D-lambda1/(1+R)+lambda2*(R/(2*sigma^2*h)+log(sigma*sqrt(2*pi)))` |

- Main dùng Taylor cho inverse thời gian và prior theo tọa độ thời gian tương đối; chi phí affine đưa được vào FlashSinkhorn.
- Journal giữ nguyên inverse `1/(1+R)` và Gaussian prior theo khoảng cách vuông góc.
- FlashOPW dùng **literal Eq.19 của main** làm score; journal dùng **`<P,D>`**.
- Hai phương pháp khác cost, prior, tham số và score: không yêu cầu cùng ranking; không quy toàn bộ chênh lệch cho Taylor.
- **Không chuyển đổi sigma journal để ép prior giống main.** Sigma của hai phương pháp có đơn vị/định nghĩa khác nhau.
- Dense affine OPW là đối chứng cùng công thức main, không phải OPW gốc journal.

## 3. Sáu nhóm experiment

Kế hoạch chi tiết: [docs/opw_experiment_plan.md](docs/opw_experiment_plan.md).

| Nhóm | Nội dung | Trạng thái tại bàn giao |
|---|---|---|
| 1 | Correctness: Flash so với dense cùng công thức, precision và hội tụ | Đã chạy; parity tốt |
| 2 | Ablation công thức, prior, Taylor và score | Đã chạy và kiểm tra |
| 3 | Tuning chỉ trên TRAIN, đóng băng riêng từng metric | Đã chạy GPU và audit |
| 4 | Toàn bộ official TEST, MAP/ACC và thống kê theo query | **Đang chạy trên server** |
| 5 | Tốc độ/bộ nhớ theo độ dài chuỗi, cùng số vòng và cùng độ hội tụ | Chưa hoàn thành protocol mới |
| 6 | Benchmark k-NN end-to-end | Chưa hoàn thành protocol mới |

Benchmark scaling cũ chưa đủ thay nhóm 5 mới: số vòng cố định khác nhau và chưa áp dụng đầy đủ tham số đóng băng riêng từng metric.

## 4. Kết quả đã biết

### Nhóm 1

- Flash FP32 so với dense/reference cùng công thức có coupling relL2 khoảng `1e-6` với default, khoảng `1e-5` với tuned.
- Flash/dense NN agreement trong các kiểm tra GPU đã gửi là 100%.
- Với 224 cặp TRAIN trong sweep: default đạt marginal L1 <=`1e-3` cho mọi cặp ở 200 vòng; tuned đạt cho mọi cặp ở mốc 1000 vòng của sweep thưa.
- Ranking ổn định sớm không chứng minh coupling đã hội tụ.

### Nhóm 2

- Các kiểm tra GPU/CPU đã audit có parity tốt.
- Khi giữ cùng prior tương đối, Taylor ở profile tuned gây sai khác coupling nhỏ và chưa làm đổi MAP/ACC trong pilot.
- Kết luận này không chứng minh main tương đương OPW journal vì prior journal khác.

### Nhóm 3

- Dataset **FacesUCR**, ba split chỉ từ TRAIN; không dùng TEST để chọn tham số.
- 11 metric, tối đa 6 candidate cho mỗi metric có tuning.
- Báo cáo nhỏ: `reports/opw_group3_gpu_20261007/`.
- File đóng băng quan trọng: **`reports/opw_group3_gpu_20261007/selected_all_metrics.json`**.
- Full kết quả gốc local: `outputs/opw_group3_gpu_import_20261007/opw_group3_gpu_20261006/`.

| Metric | Tham số selected | TRAIN ACC@1 | TRAIN MAP |
|---|---|---:|---:|
| FlashOPW main | lambda1=1, lambda2=0.1, sigma≈0.03194382825 | 79.762% | 74.405% |
| OPW journal | lambda1=1, lambda2=0.3, sigma=5 | 76.190% | 71.397% |
| TLp | weight=50, epsilon=0.1 | 80.952% | 75.130% |

Tất cả `cost_scale=1`. Đây là TRAIN sau tuning, chưa chứng minh ưu thế trên TEST.

### Kiểm tra main-vs-journal riêng

- Runner: `experiments/opw_main_vs_journal.py`; script: `scripts/run_opw_main_vs_journal.sh`.
- Báo cáo đã kiểm tra trước lần chạy fresh: `reports/opw_main_vs_journal_20261007/`.
- Log fresh GPU người dùng gửi: residual stopping cho FlashOPW 79.762% ACC / 74.405% MAP; OPW 76.190% / 71.397%; cả hai không có cặp vượt tolerance trong pilot.
- Fixed 200 vòng: FlashOPW vẫn giữ ACC nhưng **1861/2352 cặp chưa đạt tolerance**; OPW có 0 cặp.
- Nhóm 4 vì vậy dùng residual stopping, không dùng fixed 200.

## 5. Nhóm 4 đang chạy

File chính:

- `experiments/opw_group4.py`
- `experiments/opw_group4_audit.py`
- `experiments/opw_group4_resume.py`
- `scripts/run_opw_group4.sh`
- `tests/test_opw_group4.py`
- [docs/opw_group4.md](docs/opw_group4.md)

Protocol đóng băng:

- Full official TRAIN gallery: **200 chuỗi**; full official TEST: **2050 query**.
- Mỗi cấu hình: **410.000 cặp**.
- Marginal tolerance `tau=0.001`; kiểm tra mỗi 50 vòng; cap 4000; schedule `f_then_g`.
- MAP trên toàn gallery; ACC@k với `k=1,3,5,7,15,30`; ACC@1 là tiêu chí chính.
- Không tuning trên TEST, không chọn best-k theo TEST, không loại cặp chưa hội tụ.
- Flash GPU FP32; các entropic baseline dense GPU FP32. DTW/Soft-DTW/OT dùng CPU FP64 ngay cả với `--device cuda`.
- Paired bootstrap CI cho chênh lệch MAP, exact McNemar cho ACC@1, Holm correction cho các baseline trong từng profile.
- Query CI có điều kiện theo gallery/tham số cố định; không phải phương sai do retraining.

Lệnh resume đang sử dụng:

```bash
cd /home/doanpt/minh.nd/flash-opw
conda activate minh

bash scripts/run_opw_group4.sh 1 \
  --output outputs/opw_group4_facesucr_gpu_20261007 \
  --resume
```

**Tiến độ gần nhất người dùng gửi: `1184/2050`, `chunks 297/8721`.** Đây là khoảng 58% cấu hình đầu tiên (FlashOPW selected), khoảng 3.4% workload 17 cấu hình; chưa có full TEST hoàn chỉnh được gửi.

Thứ tự chính xác:

| STT | Metric | Profile | Tham số, ngoài cost_scale=1 |
|---:|---|---|---|
| 1 | FlashOPW main | selected | lambda1=1, lambda2=0.1, sigma=0.031943828249996996 |
| 2 | OPW journal | selected | lambda1=1, lambda2=0.3, sigma=5 |
| 3 | TLp | selected = preset | weight=50, epsilon=0.1 |
| 4 | OPW-KL | selected | lambda1=0, lambda2=0.3, sigma=5 |
| 5 | Sinkhorn | selected | epsilon=0.01 |
| 6 | TCOT | selected | lambda=3 |
| 7 | DTW | selected = preset | Không có tham số tuning |
| 8 | LDTW | selected = preset | Không có tham số tuning |
| 9 | NDTW | selected = preset | Không có tham số tuning |
| 10 | Soft-DTW | selected | gamma=1 |
| 11 | OT | selected = preset | OT không regularization |
| 12 | FlashOPW main | preset | lambda1=1, lambda2=0.1, sigma=1 |
| 13 | OPW journal | preset | lambda1=1, lambda2=0.1, sigma=1 |
| 14 | OPW-KL | preset | lambda1=0, lambda2=0.1, sigma=1 |
| 15 | Sinkhorn | preset | epsilon=0.1 |
| 16 | TCOT | preset | lambda=1 |
| 17 | Soft-DTW | preset | gamma=0.1 |

11 metric × hai profile = 22 trường hợp; năm metric có hai bộ trùng nhau nên còn **17 cấu hình duy nhất**, không phải tuning thêm 17 candidate.

- Mỗi cấu hình có 513 query chunks (chunk size=4), tổng **8721 chunks**.
- `results.csv` và `evaluations.json` cập nhật sau mỗi cấu hình hoàn tất.
- Full matrices nằm trong `matrices/`; cached chunks trong `chunks/`.
- `summary.json`, `comparison_report.md` và paired statistics chỉ được runner tạo sau toàn bộ cấu hình.
- Nếu chỉ hoàn tất ba metric đầu, phải ghi rõ phạm vi partial; chưa có công cụ riêng tạo báo cáo thống kê partial.
- Sau khi full run hoàn tất, audit bằng:

```bash
python -m experiments.opw_group4_audit \
  outputs/opw_group4_facesucr_gpu_20261007
```

## 6. Lỗi resume đã sửa: giữ nguyên cache và nguồn số

Resume từng lỗi ở chunk `q000176`:

```text
query=177, gallery=92
iterations=450
residual=0.001000002492219209
ValueError: Invalid cached stopping checkpoints
```

Solver kiểm tra bằng phép giảm `P@1`, diagnostic FlashOPW tính bằng `P@[1,Y]`. Hai phép giảm FP32 có thể nằm hai phía ngưỡng 0.001.

Commit **3813c08**:

- Không coi residual hơi vượt tau là checkpoint hỏng.
- Giữ residual nguyên và đếm cặp vượt tau trong quality report.
- Vẫn kiểm tra iteration, hash, identity và dữ liệu cache.
- Không nới tolerance, không đổi thuật toán số, không loại cặp.
- Có migration giới hạn cho đúng phiên bản cache cũ; backup `environment.before_checkpoint_validator_fix.json`.
- Local regression tests sau sửa: 11 CPU tests passed; CUDA test không chạy local.
- Người dùng đã resume và tiến xa hơn vị trí lỗi.

**Không sửa các numerical/source files đang đóng băng khi nhóm 4 chạy.** Runner kiểm tra source hashes; sửa solver, runtime, metric hoặc module nhóm 3/4 có thể khiến resume bị từ chối. Migration hiện tại chỉ cho phép bản sửa reader đã xác định, không phải cho mọi thay đổi.

Các nguồn cần giữ nguyên gồm package `flashopw`, `flashsinkhorn`, và các module được hash: `opw_parameters`, `opw_tune`, `opw_scaling`, `opw_knn`, `sequence_data`, `sequence_metrics`, `sequence_metrics_torch`, `retrieval`, `runtime`, cùng module nhóm 3/4 và `paired_statistics`.

Khi triển khai nhóm 5/6, ưu tiên **thêm module/script/test mới** và tái sử dụng API, giữ nguyên nguồn đóng băng. Không chạy hai tiến trình cùng ghi vào output nhóm 4.

## 7. Thời gian và ưu tiên trước báo cáo

Ước lượng tuyến tính từ pilot TRAIN, **không phải thời gian đo full TEST**:

| Phần việc nhóm 4 | Ước lượng |
|---|---:|
| FlashOPW selected | Khoảng 4 giờ |
| OPW selected | Khoảng 24 phút |
| TLp | Khoảng 1 giờ |
| OT CPU | Khoảng 18 giờ |
| Toàn bộ 17 cấu hình | Khoảng 40 giờ |

Phân bố TEST, số vòng hội tụ và môi trường có thể thay đổi thời gian đáng kể. Không lấy tốc độ FlashOPW nhân 17 vì các baseline batch khác nhau. Chưa có elapsed time thực tế của lần chạy hiện tại để cập nhật dự báo.

Ưu tiên đã đề xuất:

1. Hoàn thành ba cấu hình đầu để có full TEST cho FlashOPW, OPW journal, TLp.
2. Nhóm 5 phiên bản giới hạn: độ dài 131/512/1024/2048/4096, d=1/13, tập trung Flash, dense affine cùng công thức, OPW journal và TLp; đo warmup riêng, cùng số vòng và cùng tolerance, tốc độ và bộ nhớ.
3. Nhóm 6 workload nhỏ, ví dụ gallery=64/query=64, đo end-to-end chuyển dữ liệu, distance matrix và ranking.
4. Chừa thời gian audit và chuẩn bị báo cáo.

Budget GPU đã tư vấn cho phiên bản giới hạn: nhóm 5 khoảng 45–120 phút, nhóm 6 khoảng 15–40 phút. Đây là **dự trù chưa đo**, chưa tính thời gian triển khai/kiểm thử runner. CPU dài cho entropic/exact OT có thể rất tốn thời gian; ưu tiên CPU DTW/Soft-DTW và case ngắn nếu cần.

Chưa có yêu cầu triển khai cụ thể nhóm 5/6 sau phần tư vấn. Khi benchmark GPU, tránh chạy đồng thời nhóm 4; có thể dừng nhóm 4 rồi resume từ cache.

## 8. Giới hạn kết luận hiện tại

Có bằng chứng correctness và tuning TRAIN. Chưa thể gọi là tái hiện đầy đủ thí nghiệm journal hoặc chứng minh FlashOPW tốt hơn trên official TEST khi chưa kiểm tra kết quả full TEST.

Các bước tiếp theo phải bảo toàn cache nhóm 4, dùng đúng OPW journal gốc, giữ tham số TRAIN đóng băng và ghi rõ phạm vi hoàn thành trước deadline.

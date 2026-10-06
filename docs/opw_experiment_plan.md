# Kế hoạch experiment FlashOPW: nhóm 1–6

Ngày lập: 06/10/2026. Công thức chính giữ theo `main (2).pdf`: cost affine
với relative time `i/N,j/M`, epsilon=lambda2, score Eq.19. FlashSinkhorn và
FlashOPW tiếp tục là hai package riêng. Các thay đổi score hoặc inverse exact
chỉ là đối chứng có tên riêng, không thay định nghĩa FlashOPW chính.

Log FacesUCR 64 gallery × 64 query hiện có: mặc định ACC@1=70.312%,
MAP=55.490%; tuned ACC@1=85.938%, MAP=72.872%; TLp ACC@1=89.062%, MAP=74.911%.
Đây là kết quả ở số vòng cố định, chưa xác nhận hội tụ hoặc lợi thế cùng GPU.

## 1. Correctness và hội tụ

**Câu hỏi A:** kernel Flash có tính cùng kết quả với affine dense FP64 khi
input, cost, tham số, schedule và số vòng giống nhau không?

* Dùng input được làm tròn FP32 rồi đưa chính giá trị đó vào cả hai backend,
  để tách lỗi backend khỏi việc lượng tử hóa dữ liệu.
* So Eq.19, spatial cost, coupling P, P@[1,Y], mass, marginal hàng/cột.
  Không so trực tiếp potential khác gauge. Đối chứng là dense của công thức
  main, không phải OPW journal.
* Case gồm dữ liệu thật, chuỗi vuông/rectangular, d1/13/65/390 và prior mạnh.
* Trên gallery nhỏ, so nearest neighbor và ranking của toàn ma trận score.
  Không đòi mọi ranking phải giống nếu score margins nhỏ hơn lỗi số học;
  ghi margins và disagreement để đánh giá ảnh hưởng thực tế.

**Câu hỏi B:** 200 vòng có đủ giải và ổn định kết quả không?

* Quét 20,100,200,500,1000,2000 vòng; giữ tham số, input và score cố định.
* Ghi max(row_L1,col_L1), mass, score drift và tỷ lệ NN/ranking thay đổi.
  Đề xuất kiểm tra tau=1e-3, thêm tau=1e-4; đây là ngưỡng thí nghiệm,
  không phải một bảo đảm sai số objective phổ quát.
* Ghi tỷ lệ đạt ngưỡng ở từng checkpoint, checkpoint đầu tiên đạt, và những
  cặp chưa đạt ở cap. Không gọi 2000 vòng là nghiệm hội tụ nếu residual còn lớn.
* Khi marginal chưa khả thi, primal-minus-dual chỉ là diagnostic, không phải
  certified gap. Việc mass≈1 không bảo đảm đúng từng marginal.
* Phân biệt ba kết quả: lỗi backend; backend khớp nhưng chưa hội tụ;
  backend khớp và đã đạt tiêu chí ổn định đã khai báo.

Thực hiện trên TRAIN và synthetic, không tìm tham số theo nhãn TEST.
CPU local chỉ xác nhận oracle/precision/convergence; Flash CUDA phải chạy server.
Runner: `experiments.opw_group1`; lệnh server và kết quả local được ghi trong
report nhóm 1. Đây là diagnostic sweep, không sửa `n_iters` trong file tuning
hoặc tự chọn số vòng theo ACC test.

Lệnh thực hiện nhóm1 trên GPU1 đã được cấp, sau khi pull code và activate `minh`:

```bash
bash scripts/validate_flash_opw.sh 1
bash scripts/run_opw_group1.sh 1 \
  --flash-parameters reports/opw_tuning_20261006/selected_parameters.json \
  --output outputs/opw_group1_gpu_v1
```

Mặc định 14 query ×16 gallery của TRAIN (224 cặp mỗi preset), 6 synthetic
shape gồm131×131,d1;257×513,d13;1025×1537,d1 và feature65/390. Parity ở20/200
vòng và thêm4 cặp thật; convergence ở20/100/200/500/1000/2000. CPU mode kiểm
tra denseFP32, thêm tiledFP64 ở shape nhỏ; CUDA mode kiểm tra Flash IEEE FP32
và toàn matrix score ở mỗi checkpoint. Thêm `--profiles default tuned stress`
nếu cần kiểm tra stress mu430,epsilon.03 bằng output mới.

Output: `parity.json`, `convergence.csv`, `convergence.json`, `summary.json`,
`environment.json`, `run_state.json` và NPZ snapshots có chỉ số training.
Gửi nguyên folder để audit lại. Resume cùng lệnh với `--resume`; code, data,
selection và settings phải giữ nguyên. CPU/GPU luôn dùng output khác nhau.

## 2. Ablation prior, score, Taylor

* So mu1.05 và mu50, giữ epsilon, score và số vòng/độ hội tụ.
* Đánh giá Eq.19, <P,D>, <P,D+muF> từ cùng một coupling.
* So affine/Taylor với inverse exact, giữ relative prior, lambda1, epsilon,
  sigma, score và schedule. Tách riêng đối chứng normalized prior của journal.
* Ưu tiên TLp: preset TLp và tuned Flash cùng cost D+50F,epsilon.1 nhưng
  khác score và iterations; không quy chênh lệch accuracy cho Taylor ngay.
* Mỗi đối chứng chỉ thay một yếu tố. Score chính vẫn Eq.19; nếu chọn score
  mới cho sản phẩm, phải chọn bằng validation và báo cáo như phiên bản riêng.

## 3. Tuning công bằng

* Cùng ba training-validation split/seed, không dùng TEST để chọn.
* Flash: mu/epsilon; TLp: temporal weight/epsilon; OPW và OPW-KL: regularization
  tương ứng; Sinkhorn/TCOT/Soft-DTW: entropy/smoothing. DTW chuẩn không có
  tham số smoothing; không đổi baseline sang windowed DTW mà vẫn giữ cùng tên.
* Ngân sách search tương đương, ghi rõ số candidate, ranges và thời gian search.
  ACC@1 ưu tiên, MAP phá hòa; tiêu chí khai báo trước.
* Freeze trước test. Hai bảng riêng: preset journal và các metric đều được
  tuning trên train. Ghi iteration policy đã xác định từ nhóm 1.

## 4. MAP/ACC trên dữ liệu thật

* FacesUCR đầy đủ: gallery200,query2050; FaceAll: gallery560,query1690.
  Cả hai dài131,d1 nên chưa đại diện native long sequences.
* Sau đó thêm dữ liệu thật dài/variable-length và nhiều feature nếu có dữ liệu
  và preprocessing thích hợp (ví dụ các feature SAD/Action3D trong journal).
* Cùng split, feature, preprocessing, gallery/query và ground-cost convention.
  Mọi preprocessing học từ dữ liệu phải fit trên TRAIN.
* ACC tại k1,3,5,7,15,30; k chính được khai báo trước. MAP trên toàn gallery,
  không chọn gallery hoặc k theo kết quả TEST.
* Paired bootstrap 95% CI cho MAP differences; McNemar cho đúng/sai ACC trên
  cùng query. Dùng lại distance matrices cho các k và thống kê, không solve lại.
* Báo cáo riêng variability do ba training selections và uncertainty theo query.
  Không coi chênh lệch hai query ở pilot64 là bằng chứng ưu thế ổn định.

## 5. Tốc độ và bộ nhớ cùng GPU

* Flash vs affine dense: cùng GPU,FP32,input,cost,parameters,schedule,iterations.
  Đây là phép so trực tiếp engine.
* Thêm dense GPU Sinkhorn,TLp,OPW,OPW-KL,TCOT; reference implementations
  của repo, không phải code tác giả. Giữ score riêng của từng metric.
* Hai chế độ: cùng200 vòng và thời gian đến cùng tiêu chí marginal. Cùng số
  vòng không đồng nghĩa cùng độ hội tụ; ghi tỷ lệ cap/nonconvergence.
* Length128,256,512,1024,2048,4096,8192; bắt đầu d1/13/64, mở rộng390 sau.
  N=M và M≈1.25N; ba seed, warmup ngoài timing, ít nhất năm repeat nếu ngân sách cho phép.
* Đo median/dispersion, thời gian mỗi cặp, CUDA peak extra memory; khai báo
  input resident hay có truyền dữ liệu. Diagnostics ngoài phần timed.
* CPU DTW/LP vẫn là đối chứng thực tế, nhưng tỉ số CPU/GPU ghi đúng backend.
  Memory/resource skip không có timing giả hoặc speedup vô hạn.
* Synthetic phục vụ scaling; resampling không biến một bộ ngắn thành native
  long dataset của journal. Các phép đo accuracy vẫn dùng dữ liệu thật.

## 6. Hiệu năng k-NN toàn quy trình

* Đo distance matrix, chuyển dữ liệu, ranking/evaluation và tổng wall time.
* So pair-by-pair với batching khi có implementation đúng và đã kiểm tra parity.
* Cùng workload/gallery/query, resource allocation và các lựa chọn accuracy
  đã freeze. Kết hợp bảng accuracy–time–memory để chọn tradeoff ứng dụng.
* Phân biệt kernel timing, pair latency, matrix throughput và thời gian toàn
  ứng dụng. Record phiên bản, hashes, hardware và tải tại thời điểm đo.

## Thứ tự và ngân sách

Nhóm1 → nhóm2 → nhóm3 → nhóm5 pilot → nhóm4 đầy đủ → nhóm6.
Từ log hiện tại, ngoại suy giữ nguyên backend và pair latency: mỗi Flash
variant đầy đủ FacesUCR khoảng2.7 giờ tính distance; đủ12 metric khoảng64 giờ.
Đây là estimate, không phải runtime đã đo. Chạy pilot và ước lượng ngân sách
trước khi mở full split; ưu tiên baseline quan trọng, caching và batching đã
kiểm tra correctness. Chưa gọi là tái hiện journal nếu feature, split, cost,
parameters hoặc evaluation protocol còn khác.

Trạng thái 06/10/2026: nhóm1 đã hoàn tất phần CPU (52/52 parity checks đạt;
224 cặp mỗi cấu hình, sweep đến 2000 vòng). Đã nhận console log Flash CUDA:
40/40 parity case đạt; 12 matrix có NN agreement 100% với FP64 cùng số vòng.
CSV GPU đã xác nhận hội tụ pilot: tất cả224 cặp default đạt tau=1e-3 tại200,
tuned tại1000; tau=1e-4 đạt toàn bộ tại500(default),2000(tuned).
Còn chờ summary GPU để xác nhận matrix score parity và tổng trạng thái run.
Xem [báo cáo nhóm1](../reports/opw_group1_20261006/review.md) và artifact đã lưu.
Nhóm2–6 là kế hoạch.
Runner hiện có: correctness tests, tuning Flash, k-NN cố định, GPU scaling
ở số vòng cố định. Tuning các baseline, statistical analysis và throughput
batch còn cần bổ sung khi thực hiện các nhóm tương ứng.

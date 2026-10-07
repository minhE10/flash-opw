# Nhóm 4: MAP/ACC trên toàn bộ TEST

Runner `experiments.opw_group4` dùng file freeze **tất cả metric** của nhóm 3,
không dùng schema Flash-only cũ. Không chọn tham số, score, k hoặc cap bằng TEST.
Dataset được lấy từ file freeze; FacesUCR dùng toàn bộ 200 TRAIN làm gallery
và 2050 TEST làm query, không balanced subset. Có thể dùng FaceAll (560/1690)
hoặc NPZ khác khi có artifact TRAIN tương ứng.

## Protocol đã khai báo

* Hai bảng **selected** và **preset**, đủ 11 metric: FlashOPW, OPW gốc, TLp,
  OPW-KL, Sinkhorn, TCOT, DTW, LDTW, NDTW, Soft-DTW và OT.
* FlashOPW giữ main Eq.19. OPW gốc giữ inverse exact `1/(1+R)`, Gaussian
  perpendicular `R/(1/N^2+1/M^2)`, sigma trong đơn vị journal và score `<P,D>`.
  Không dùng dense affine hoặc inverse-relative thay OPW gốc.
* Dữ liệu được làm tròn FP32 một lần; CPU FP64 và GPU FP32 nhận cùng giá trị.
  Giữ feature, thứ tự mẫu và squared-L2 ground cost như nhóm 3, không normalize
  bằng TEST. Đây là protocol repo hiện tại, không tự động tái hiện mọi bảng journal.
* Entropic methods dùng policy đã freeze: marginal hàng và cột L1 <= 1e-3,
  kiểm tra mỗi 50 vòng, cap 4000, `f_then_g`. Không thay bằng fixed200: lần
  kiểm tra main/journal mới cho thấy 1861/2352 cặp Flash chưa đạt tolerance ở200.
* MAP trên **toàn gallery**, cùng class là relevant. ACC tại k=1,3,5,7,15,30;
  **ACC@1 là chính**. Majority vote hòa chọn class gần nhất; distance hòa giữ
  gallery index. Không chọn best-k theo TEST, không bỏ query/cặp khó.
* Ưu tiên phân tích FlashOPW so với OPW gốc và TLp, các metric còn lại là
  secondary. Mỗi phép so dùng đúng các query tương ứng của hai phương pháp.
* MAP difference: paired query bootstrap percentile, 10000 resample, seed
  20261007, CI95% chưa hiệu chỉnh nhiều phép so. Query effects được giữ ghép
  cặp; gallery và tham số được giữ cố định. [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
  mô tả resampling và phương pháp percentile.
* ACC@1: McNemar exact hai phía bằng binomial test trên hai nhóm discordant,
  null p=0.5. Không có discordant thì p=1. API dùng là
  [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
  Báo cả raw p và Holm-adjusted p cho toàn bộ baseline được yêu cầu trong mỗi
  profile; mặc định 10 phép so/profile. Không điều chỉnh CI MAP bằng Holm.
* Các CI/p-value giả định query độc lập, có điều kiện theo gallery/selection
  hiện có. Không thay cho uncertainty do subject/group phụ thuộc hoặc retraining.
  Nhóm 3 hiện freeze **một** bộ tham số/metric bằng objective tổng hợp trên ba
  holdout TRAIN, không có ba selection độc lập. Không gán SD của nhóm 3 cho TEST.
  Muốn phân tích training-selection variability cần các artifact TRAIN độc lập
  được freeze trước; không tạo chúng bằng việc nhìn kết quả TEST.

## Lệnh trên server

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m pytest tests/test_opw_group4.py --require-gpu -q

bash scripts/run_opw_group4.sh 1 \
  --selection reports/opw_group3_gpu_20261007/selected_all_metrics.json \
  --output outputs/opw_group4_facesucr_gpu_20261007
```

Lệnh chính dùng toàn bộ split, cả selected và preset, mặc định tất cả 11 metric.
Không truyền `--fixed-iters` hay chỉnh policy. GPU1 là allocation đã dùng trước;
wrapper giữ allocation, memory fraction0.45 và hai CPU thread. Flash/dense entropic
chạy GPU; DTW/Soft-DTW/LP OT chạy CPU. CPU mode là kiểm tra FP64, không được gọi
là benchmark Flash GPU. Thời gian solve trong report không phải benchmark nhóm5.

Full FacesUCR có 410000 cặp/configuration. Preset và selected trùng tham số
trong cùng metric dùng chung ma trận. Với artifact hiện tại còn **17 configuration
riêng**, tổng 6970000 cặp cần tính. Không re-run tuning. Chạy đầy đủ có thể lâu,
đặc biệt CPU LP OT; chưa có runtime full split để khẳng định số giờ.
Để tránh mất session terminal, có thể chạy cùng lệnh bằng `nohup`:

```bash
mkdir -p outputs
nohup bash scripts/run_opw_group4.sh 1 \
  --selection reports/opw_group3_gpu_20261007/selected_all_metrics.json \
  --output outputs/opw_group4_facesucr_gpu_20261007 \
  > outputs/opw_group4_facesucr_gpu_20261007.log 2>&1 &
tail -f outputs/opw_group4_facesucr_gpu_20261007.log
```

Chọn một cách khởi chạy; không chạy đồng thời hai process vào cùng output.
Runner checkpoint sau mỗi4 query; khi bị ngắt:

```bash
bash scripts/run_opw_group4.sh 1 \
  --selection reports/opw_group3_gpu_20261007/selected_all_metrics.json \
  --output outputs/opw_group4_facesucr_gpu_20261007 --resume
```

Resume phải giữ nguyên source, dữ liệu TRAIN/TEST, file freeze, options và
environment. Chunk đã lưu được kiểm tra SHA256/indices/labels/checkpoint trước
khi dùng lại; chunk dang dở được tính lại. Chỉ chunk cuối chưa lưu mới có thể mất.
Không sửa source/pull commit mới khi đang chạy nếu muốn resume cùng signature.

### Bản vá kiểm tra checkpoint sát ngưỡng

Log server ghi nhận query177/gallery92 dừng ở450 vòng, nhưng residual diagnostic
FP32 là `0.001000002492219209`, cao hơn tau1e-3 khoảng `2.49e-9`. Engine kiểm
tra dừng bằng `P@1`; OPW diagnostic tính hàng bằng `P@[1,Y]`. Hai reduction
có thể cho kết quả ở hai phía ngưỡng. Validator ban đầu đã nhầm điều kiện
`residual>tau` trước cap với checkpoint hỏng và dừng toàn bộ experiment.

Bản vá tách hai việc: hash/identity/shape/số vòng vẫn phải hợp lệ; residual
được giữ nguyên và dùng để đánh giá quality. Mọi residual>tau vẫn nằm trong
`unconverged_pairs`, kể cả cặp sát ngưỡng đã early-stop. Không làm tròn residual,
không nới tolerance, không đổi solver, tham số hoặc stopping schedule.

Để tiếp tục run của bản gốc commit `5eed549`:

```bash
git pull --ff-only origin main
bash scripts/run_opw_group4.sh 1 \
  --output outputs/opw_group4_facesucr_gpu_20261007 --resume
```

Runner chỉ cho phép migration từ **đúng hash bản runner gốc** sang **đúng
hash bản vá này**. Toàn bộ source tính toán khác, dữ liệu, config và environment
phải khớp. Nó lưu `environment.before_checkpoint_validator_fix.json` trước
khi cập nhật signature và ghi provenance migration; các NPZ/job đã lưu được
giữ nguyên. Chạy tiếp vẫn kiểm tra hash từng chunk. Log `[cached]` là đọc lại
chunk, `[computed]` là tính mới; progress in lại từ4 không có nghĩa solve lại.
Audit cũng kiểm tra provenance của migration. Đây là ngoại lệ riêng cho bản
vá validator, không phải tùy chọn bỏ kiểm tra source khi resume.

Nếu ngân sách chỉ đủ các đối chứng chính, khai báo trước workload nhỏ hơn
bằng `--profiles selected --metrics flash-opw opw tlp` với output **khác**.
Lệnh đó vẫn dùng toàn bộ TEST nhưng chỉ có ba metric, không phải bảng đầy đủ11.
Không đổi danh sách metric/profile giữa chừng bằng `--resume`.

## Output, hội tụ và audit

* `results.csv`: MAP/ACC từng k, backend, tham số, số cặp chưa hội tụ,
  residual và iterations. Cập nhật sau mỗi configuration hoàn tất.
* `evaluations.json`: AP và predictions theo từng query/k; tái sử dụng cùng
  ma trận, không solve thêm cho từng k hoặc thống kê.
* `paired_statistics.json/.csv`: MAP delta/CI95%, ACC@1 delta, b/c discordant,
  raw McNemar p, Holm p và convergence-valid flag.
* `comparison_report.md`, `summary.json`, `run_state.json`: bảng tổng kết và
  trạng thái, chỉ hoàn tất khi toàn bộ workload đã khai báo được tính xong.
* `environment.json`, `data_manifest.json`, `frozen_selection.json`: versions,
  sources, hardware, full indices, TRAIN/TEST fingerprints và tham số gốc.
* `chunks/`: NPZ cùng job JSON cho resume; `matrices/`: ma trận đầy đủ.
  DTW/Soft-DTW/OT có residual/iterations placeholder0, được báo **N/A**, không
  xem placeholder là chứng nhận Sinkhorn hội tụ.

Nếu có cặp entropic chưa đạt ngưỡng tại cap, giữ nguyên chúng trong MAP/ACC
và ghi `completed_with_nonconvergence`; so sánh liên quan chỉ là provisional.
Không tự tăng cap hoặc loại query sau khi xem TEST. Nonfinite score thì dừng
và ghi lỗi; không cho ma trận thiếu distance đi vào ranking.

Sau khi chạy xong, kiểm tra toàn bộ artifact mà không giải OT lại:

```bash
python -m experiments.opw_group4_audit \
  outputs/opw_group4_facesucr_gpu_20261007
```

Audit kiểm tra source/data, chunk hashes, ma trận đầy đủ với chunks, AP/predictions,
MAP/ACC, residual/iterations, bootstrap/McNemar/Holm và hai CSV/báo cáo.
Không tuyên bố audit re-solve mọi coupling hoặc đo tốc độ GPU.

Để gửi kết quả, cần các file report/JSON/CSV nêu trên; giữ chunks/matrices
nguyên vẹn trên server cho audit đầy đủ. Chưa cần commit NPZ hoặc ZIP lớn vào
repo. Có thể gửi trước `comparison_report.md`, `results.csv`,
`paired_statistics.json`, `summary.json`, `audit.json`, `environment.json`,
`data_manifest.json`, `frozen_selection.json` để đọc và đối chiếu.

## FaceAll và dữ liệu khác

Không dùng freeze FacesUCR cho FaceAll. Nếu chưa có selection FaceAll, chạy
nhóm3 trên TRAIN FaceAll với ngân sách đã khai báo rồi dùng artifact đó:

```bash
bash scripts/run_opw_group3.sh 1 --dataset FaceAll \
  --gallery 28 --queries 28 --budget 6 \
  --output outputs/opw_group3_faceall_gpu

bash scripts/run_opw_group4.sh 1 \
  --selection outputs/opw_group3_faceall_gpu/selected_all_metrics.json \
  --output outputs/opw_group4_faceall_gpu
```

NPZ khác phải có train/test arrays và labels (hoặc packed values/offsets),
cùng fingerprint TRAIN đã dùng để tune. Native long/multifeature dataset
cần preprocessing và artifact TRAIN riêng; resampling FacesUCR không làm nó
thành native long dataset của journal.

## Kiểm tra local

Các test CPU kiểm tra đủ11 metric trên bộ TRAIN/TEST synthetic nhỏ, so từng
metric với reference SciPy, kiểm tra dedup17 configuration, toàn gallery/query,
audit thống kê và ma trận, ngắt/resume, từ chối cache hỏng hoặc TEST bytes thay đổi.
Thêm GPU smoke test cho FlashOPW/OPW gốc/TLp với score oracle FP64; chạy trên
server bằng lệnh pytest ở trên. Không chạy full TEST thật trên Windows CPU.

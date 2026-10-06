# FlashOPW: chọn tham số và đo chuỗi dài

Giữ công thức affine, tọa độ thời gian `i/N`, `j/M` và score Eq.19 của
`main (2).pdf`. Không đổi sang prior vuông góc của journal hoặc inverse OPW.
`flashsinkhorn/` tiếp tục là engine riêng; thay đổi nằm ở tầng experiment.

## Chọn tham số bằng train

`experiments.opw_tune` chỉ đọc TRAIN. Với NPZ, tuner không cần và không mở
`test_x`, `test_values` hay `test_labels`. Mỗi class được chia thành gallery
và validation riêng, không có cùng sample ở hai phần; class có dưới hai mẫu
bị từ chối. Lưu chỉ số gốc, seed và SHA256 nội dung train để kiểm tra lại.
Sau khi chọn xong, gallery của phép đánh giá test được phép dùng lại toàn bộ
train hoặc subset train đã khai báo.

Grid mặc định gồm 18 cặp `mu × lambda2`:

| Tham số | Giá trị |
|---|---|
| `mu` | 0.1, 1.05, 10, 50, 200, 430 |
| `lambda2` / epsilon | 0.03, 0.1, 0.3 |
| score | Eq.19 (`pdf-loss`) |
| iterations | 200, f rồi g |

Tuner luôn thêm bộ mặc định của dataset, loại các cặp `mu,lambda2` trùng.
FacesUCR có 18 ứng viên; FaceAll có 19 vì mặc định `mu=10.05` chưa nằm trong
grid. Đây là phạm vi search thực dụng, chưa phải kết luận các giá trị này tối ưu.
Không tăng iterations hoặc đổi score theo kết quả trên TEST.

Trong cost affine, `mu=lambda1+lambda2/(2*sigma²)` quyết định temporal cost.
Với cùng `mu` và epsilon, nhiều cặp `lambda1,sigma` tạo cùng coupling và
chỉ khác hằng `q0`, nên cho cùng ranking. Tìm trên hai tham số hiệu dụng
giúp tiết kiệm các lượt chạy trùng. Tuner giữ `lambda1` mặc định khi `mu`
lớn hơn nó; nếu cần `mu` nhỏ hơn, dùng `lambda1=mu/2`, rồi suy ra `sigma`.
Cost scale giữ nguyên. Trên FacesUCR, `mu=50,epsilon=.1,lambda1=1` tương ứng
`sigma≈.03194`, mạnh hơn đáng kể so với sigma=1.

Mặc định ưu tiên ACC@1, dùng MAP để phá hòa; hòa cả hai thì giữ ứng viên
xuất hiện trước, bộ mặc định đứng đầu. Dùng `--objective MAP` để đảo thứ tự
ưu tiên hoặc `--selection-k 3` để tối ưu ACC@3. Chọn mục tiêu trước khi chạy.
Không đảm bảo cả ACC lẫn MAP test đều tăng. Với validation nhỏ, cần kiểm tra
lại trên nhiều seed/split được khai báo trước khi đưa ra kết luận tổng quát.

`selected_parameters.json` đóng băng bộ được chọn. k-NN dùng file này cho
`flash-opw-tuned`; `flash-opw` và tất cả baseline journal giữ bộ tham số cũ.
`affine-opw-dense-tuned` là đối chứng FP64 tùy chọn. Runner từ chối nếu dataset,
train fingerprint, score hoặc iterations khác file đã đóng băng. Phiên bản và
source hashes được lưu trong metadata; nội dung selection cũng tham gia resume.

Sai số marginal được ghi riêng: CPU oracle đo mọi cặp validation, GPU đo
8 cặp của query đầu cho mỗi ứng viên. Temporal prior mạnh có thể chưa hội tụ
sau 200 vòng; accuracy tốt hơn ở số vòng cố định không chứng minh đã giải OT
hội tụ. Hãy đọc diagnostics cùng với MAP/ACC và kiểm tra parity của bản tuned.

## Chạy trên server

Chạy tuần tự trên GPU 1 đã được cấp. Tuner mặc định có 32 gallery, 28 validation
và 18 ứng viên (16.128 cặp trên FacesUCR). Để thử nhanh trước, dùng
`--gallery 28 --validation 14 --mus 1.05 50 430 --epsilons .1` và output riêng.

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
python -m pip install -e '.[dev,plots,opw]'
bash scripts/validate_flash_opw.sh 1

bash scripts/run_opw_tuning.sh 1 --dataset FacesUCR \
  --output outputs/opw_tune_facesucr_v1

bash scripts/run_opw_knn.sh 1 --datasets FacesUCR \
  --flash-parameters outputs/opw_tune_facesucr_v1/selected_parameters.json \
  --metrics flash-opw dtw ldtw ndtw soft-dtw ot sinkhorn tlp opw-kl tcot opw \
  --max-train 64 --max-queries 64 --seed 20261007 \
  --output outputs/opw_knn_tuned_facesucr_v1

bash scripts/run_opw_scaling.sh 1 \
  --flash-parameters outputs/opw_tune_facesucr_v1/selected_parameters.json \
  --lengths 128 256 512 1024 2048 4096 8192 \
  --dimensions 1 13 --seeds 42 43 --repeats 3 --equal-entropic-iters 200 \
  --metrics flash-opw affine-opw-dense-gpu sinkhorn-dense-gpu tlp-dense-gpu \
    opw-kl-dense-gpu tcot-dense-gpu opw-dense-gpu dtw ldtw ndtw soft-dtw ot \
  --output outputs/opw_scaling_v1
```

Lệnh k-NN tự thêm `flash-opw-tuned`. Đối chứng dense được tính cho các cặp
diagnostic; muốn so sánh toàn bộ ranking, thêm `affine-opw-dense` và
`affine-opw-dense-tuned` vào `--metrics` (sẽ tăng thời gian CPU).
64×64×13 metric là thử nghiệm rộng hơn pilot 32×16, chưa phải toàn bộ journal.
Để đánh giá toàn bộ FacesUCR, dùng `--max-train 0 --max-queries 0`; nên chạy
Flash mặc định/tuned trước, vì baseline CPU đặc biệt exact OT tốn thời gian.
Ví dụ `--metrics flash-opw --max-train 0 --max-queries 0` vẫn tự thêm bản tuned.

Để xác nhận nhanh bộ tham số của CPU pilot mà không chạy lại search, thay
đường dẫn `--flash-parameters` bằng
`reports/opw_tuning_20261006/selected_parameters.json`, dùng output mới.
[Báo cáo pilot](../reports/opw_tuning_20261006/review.md) ghi rõ chỉ có 3 ứng viên,
28 query test và mức sai số marginal; không mặc định rằng grid 18 cho cùng lựa chọn.

Mọi runner có checkpoint và `--resume`: dùng lại đúng lệnh và cùng output,
thêm `--resume` nếu bị gián đoạn. Resume từ chối thay đổi code, tham số, dữ liệu
hoặc môi trường. Scaling giữ cả hàng skipped/timeout khi resume; muốn chạy lại
với ngân sách khác cần output mới. Không cập nhật code giữa các bước của một run.

## Đo tốc độ chuỗi dài

Benchmark tạo hai chuỗi sin liên tục, warp thời gian 1.15, noise 0.03; chia
feature cho sqrt(d) để spatial scale tương đương giữa các dimension. Giá trị
được làm tròn FP32 trước khi đưa cả CPU và GPU cùng input. Mặc định `m=1.25n`
để có chuỗi độ dài khác nhau; dùng `--length-ratio 1` để kiểm tra trường hợp vuông.
Đây là dữ liệu tổng hợp phục vụ scaling; không dùng nó báo MAP/ACC journal.
Dữ liệu thật có chuỗi dài vẫn có thể đánh giá k-NN bằng `--dataset-file`:
NPZ fixed hoặc packed đã mô tả trong [flash_opw.md](flash_opw.md).

* FlashOPW và `affine-opw-dense-gpu` chạy trên cùng GPU, FP32, cùng cost,
  cùng tham số và số vòng. Tỉ số thời gian ở đây là đối chứng trực tiếp của engine.
* Có thêm `sinkhorn-dense-gpu`, `tlp-dense-gpu`, `opw-dense-gpu`,
  `opw-kl-dense-gpu`, `tcot-dense-gpu`: reference Torch FP32 trên cùng GPU,
  được đối chiếu với NumPy/SciPy FP64. Đây là implementation reference của repo
  này, không phải code tác giả. Cost/score của từng metric vẫn khác nhau.
  Mặc định giữ iterations journal; `--equal-entropic-iters 200` đặt cùng
  200 vòng cho tất cả metric entropic, gồm cả CPU. Chỉ dùng file selection
  có n_iters tương ứng. Chưa thể gọi đây là so sánh tại cùng độ hội tụ;
  diagnostics GPU có row/column residual cho từng phương pháp.
* Mười metric journal dùng reference CPU FP64 hiện có. Tỉ số CPU/GPU là
  thời gian triển khai thực tế, có khác backend và số vòng; không dùng nó
  kết luận Flash tăng tốc thuật toán journal bao nhiêu lần.
* Một warmup cho từng case không tính vào thời gian. Đo đầy đủ một cặp, gồm
  tạo cost/temporal features, solve và đọc scalar; input đã resident. Ghi
  từng repeat, median/min/max, seed và hashes input. Không tính diagnostic.
* Đo peak bộ nhớ allocator CUDA tăng thêm, loại input đã resident và context;
  không gọi đây là toàn bộ VRAM của GPU. Kiểm tra VRAM còn trống trước dense.
* Exact OT LP giới hạn mặc định 65.536 entries; do đó ở ratio1.25, n256 đã
  vượt giới hạn. Để có điểm LP n256, dùng ratio1 hoặc tăng giới hạn chủ động.
  CPU dense có 512MiB working estimate, GPU dense 256MiB và kiểm tra VRAM.
  CPU baseline được chạy trong child process riêng, deadline mặc định 45s
  cho warmup + repeats; timeout chỉ dừng child của benchmark này.
* Skipped, timeout, failed đều được ghi với lý do; không gán thời gian vô hạn
  hay tính speedup trên điểm thiếu. Budget là estimate working arrays, không
  phải hard limit RSS. Code lỗi thật trả exit1; skipped/timeout đã khai báo
  không bị giấu khỏi `run_state.json`.

Mở rộng feature sau lượt đầu bằng output khác, ví dụ
`--dimensions 64 390 --lengths 256 512 1024 2048 --seeds 42 43`.

Output gửi lại để phân tích:

* Tuning: `search.json`, `selected_parameters.json`, `validation.csv`, `environment.json`.
* k-NN: toàn folder output (score matrices, labels, MAP/ACC, parity, diagnostics).
* Scaling: `timings.csv`, `timings.json`, `checks.json`, `comparisons.json`,
  `run_state.json`, `environment.json`, `scaling_d*.png`.

Có thể gom cả ba folder bằng tar như lần pilot trước. Các kết quả CPU cục bộ
và GPU server phải ở output khác nhau, không dùng CPU thay cho Flash timing.

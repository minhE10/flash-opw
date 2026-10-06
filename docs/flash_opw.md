# FlashOPW theo `main (2).pdf`

`main (2).pdf` (bản ngày 02/10/2026, 3 trang) là chuẩn cho **phép sửa
chi phí và loss FlashOPW**. `OWD_journal.pdf` chỉ cung cấp các baseline,
split dữ liệu và cách đánh giá. Hai mô hình được giữ riêng:

- `flashsinkhorn`: lõi Sinkhorn, transport, gradient và HVP đã được kiểm tra
  trong các experiment trước. Đây vẫn là implementation tự reproduce.
- `flashopw`: phép mở rộng đặc trưng theo `main`, gọi lõi `flashsinkhorn`.
  Các tên biến môi trường `FLASHOPW_*` cũ của kernel vẫn được giữ để các
  lệnh benchmark FlashSinkhorn trước đây tiếp tục hoạt động.

## Ánh xạ công thức vào code

Với chỉ số **bắt đầu từ 1**, đặt `t_i=i/N`, `s_j=j/M`,
`F_ij=(t_i-s_j)^2`. Mặc định `D_ij=||x_i-y_j||²`.

| Công thức trong main | Implementation |
| --- | --- |
| (5), (7): `mu=lambda1+lambda2/(2*sigma²)` | `OPWParameters.mu` |
| (5): `q0=lambda2*log(sigma*sqrt(2*pi))-lambda1` | `OPWParameters.q0` |
| (11): `[x_i, sqrt(mu)*t_i]` | `temporal_features` |
| (12): `C_new=D+mu*F` | Khoảng cách bình phương của đặc trưng mở rộng |
| (14)–(16): `epsilon=lambda2`, bias log trọng số, cập nhật f rồi g | `flashsinkhorn.solver.sinkhorn_flash` |
| (17), (18): khôi phục thế bằng chuẩn đặc trưng mở rộng | `OPWResult.f`, `OPWResult.g` |
| (19): `sum(a*f)+sum(b*g)-lambda2+q0` | `OPWResult.loss`, score k-NN mặc định |

`q0` được bỏ khỏi vòng giải vì không thay đổi coupling; nó được cộng lại
trong loss. FlashOPW không tạo cost hoặc coupling `N x M`. Mở rộng đặc trưng
tốn `O((N+M)(d+1))`; tính loss chỉ cần tổng trọng số của thế. Giới hạn lõi
hiện tại là **1023 đặc trưng không gian + 1 tọa độ thời gian**.
`cost_scale` là tùy chọn mở rộng: không gian được nhân `sqrt(cost_scale)`;
mặc định 1 đúng theo main.

```python
from flashopw import opw_flash, opw_distance, opw_diagnostics
from flashsinkhorn import sinkhorn_flash

plain = sinkhorn_flash(x, y, epsilon=0.1)
ordered = opw_flash(x, y, lambda1=1, lambda2=0.1, sigma=1, n_iters=200)
score = ordered.loss                    # main (19)
spatial_score = opw_distance(ordered)  # <P,D>, tùy chọn đối chiếu
stats = opw_diagnostics(ordered)
```

### Phạm vi và quy ước loss

Phép Taylor `1/(1+F) ≈ 1-F` là một **xấp xỉ**. Nếu giữ cùng `F` như main,
sai số cost xấp xỉ trừ cost inverse chính xác là `lambda1*F²/(1+F)`.
FlashOPW này không mặc nhiên cho cùng nghiệm với OPW inverse ban đầu.

Mã giữ nguyên các thế và loss **đúng biểu thức (15)–(19)**. Bias log marginal
trong (15)–(16) tương ứng quy ước KL với `a⊗b` của lõi. Giá trị (19) khác
giá trị objective dùng bare entropy `sum(P*log(P))` bởi hằng số quy ước.
`entropy_f/g` và diagnostics báo riêng objective bare entropy. Không thay
loss (19) bằng objective này hoặc bằng `<P,D>` trong score mặc định.
Loss có thể âm; k-NN xếp tăng dần score và không cắt về 0.

Main ghi công thức sau hội tụ. FlashOPW mặc định 200 vòng cố định và lưu sai
số marginal; baseline OPW journal dùng preset dừng sớm 20 vòng riêng.
200 vòng cũng không đảm bảo hội tụ cho mọi dữ liệu: nếu sai số còn lớn,
tăng `--iters` trong một run mới. API hỗ trợ `n_iters=1000, tol=1e-5` để
dừng khi marginal đạt ngưỡng, nhưng trial parity mặc định dùng cùng số
vòng cố định cho FlashOPW và affine dense. `primal_minus_dual` là chẩn đoán với coupling
hiện tại; nó không chứng nhận gap tối ưu khi marginal chưa khả thi.

## Trial k-NN: MAP và ACC

Mỗi test sequence là query, training sequences là gallery. Mọi metric dùng
cùng split, cùng chỉ số subset, cùng frame features, và squared Euclidean
ground cost theo main. Danh sách gồm FlashOPW, đối chứng affine dense độc
lập FP64, và **10 metric trong journal**:

| Metric | Score dùng để xếp hạng |
| --- | --- |
| FlashOPW / affine dense | main (19), mặc định |
| DTW | Chi phí đường đi tối ưu |
| lDTW | DTW / chiều dài query |
| nDTW | DTW / số bước đường đi được chọn |
| Soft-DTW | Soft minimum đường đi, `gamma=0.1` |
| OT | Transport cost không entropy, LP HiGHS |
| Sinkhorn | `<P,D>` với `epsilon=0.1` |
| TLp (`p=2`) | `<P,D+tlp_weight*F>`, weight mặc định 50 |
| OPW-KL | `<P,D>`, OPW journal với `lambda1=0` |
| TCOT | `<P,D*(1+abs(t-s))>`, entropy `epsilon=1/tcot_lambda` |
| OPW | `<P,D>`, inverse moment và Gaussian prior của journal |

**Riêng prior OPW/OPW-KL journal** dùng khoảng cách vuông góc
`F_journal=F/(1/N²+1/M²)` theo (12) của journal. FlashOPW giữ `F` của main.
Baseline `opw-exact-relative` tùy chọn giữ inverse chính xác nhưng dùng
`F` của main, giúp tách ảnh hưởng phép Taylor và thay đổi prior.
`--opw-score spatial` đổi FlashOPW/affine dense sang `<P,D>` cho một run
đối chiếu riêng; không phải score mặc định theo main.

ACC dùng majority vote với `k=1,3,5,7,15,30`. Khi hòa phiếu, chọn class
của hàng xóm gần nhất trong các class hòa. Hòa distance giữ thứ tự gallery.
MAP tính AP từ **toàn bộ gallery đã chọn**, relevant khi cùng class;
MAP không phụ thuộc k. Query không có gallery cùng class có AP=0 và được
đếm trong output; cặp lỗi/NaN không bị âm thầm bỏ. CSV ghi cả fraction và %.

FacesUCR/FaceAll tải các file TRAIN/TEST `.ts` chính thức từ Zenodo, lưu
SHA256. Giá trị frame giữ nguyên, không resampling/normalization. Trial
mặc định 64 gallery/32 queries được chọn cân bằng class với seed 42;
`--max-train 0 --max-queries 0` dùng toàn bộ split chính thức. Với dataset
khác, cung cấp `--dataset-file` NPZ chứa features và split mong muốn.

Parameters FacesUCR: `(lambda1,lambda2,sigma)=(1,0.1,1)`;
FaceAll: `(10,0.1,1)`. Dataset khác mặc định `(50,0.1,1)` và cần đặt tham số
theo thiết kế riêng. FlashOPW/affine dense mặc định 200 vòng (`--iters`),
OPW/OPW-KL journal 20 vòng (`--journal-opw-iters`), Sinkhorn/TCOT/TLp 100 vòng.
Các solver reference cũng cập nhật f rồi g; journal Alg.2 cập nhật chiều
ngược lại. Trial không grid search theo test labels. Vì tham số, ground
cost và subset được cố định như trên, kết quả trial **không phải tái hiện
toàn bộ bảng journal**, cũng chưa đủ kết luận metric nào tốt nhất.

FlashOPW chạy CUDA FP32 với IEEE mặc định; baseline chạy CPU FP64.
Thời gian trong CSV là wall time tính ma trận score sau warmup, không bao
gồm tải dữ liệu, evaluation, diagnostics hoặc vẽ hình. Không dùng cột này
để kết luận speedup giữa các thuật toán khi backend khác nhau.

## Chạy trên server

Trong repo của bạn và môi trường `minh`, sau khi đã được cấp GPU 1:

```bash
git pull --ff-only origin main
conda activate minh
python -m pip install -e '.[dev,plots,opw]'
bash scripts/validate_flash_opw.sh 1
```

Chỉ chạy trial khi validation qua:

```bash
bash scripts/run_opw_knn.sh 1 --datasets FacesUCR \
  --max-train 32 --max-queries 16 \
  --output outputs/opw_knn_facesucr_pilot_20261006
```

32 gallery × 16 queries × 12 metric là bước đầu nhỏ để kiểm tra. Sau khi
parity, marginal và evaluation hợp lệ, có thể chạy split đầy đủ hoặc
FaceAll. Giữ nguyên allocation nếu scheduler đặt `CUDA_VISIBLE_DEVICES`.
Các script không tìm GPU trống và không tự thay allocation đang có.

Output cần gửi lại: `knn_results.csv`, `affine_parity.json`,
`diagnostics.json`, `failures.json` và `environment.json`. Folder còn lưu
distance matrices, predictions, AP từng query, dataset manifest và hình
MAP/ACC. Mỗi query xong được checkpoint; nếu gián đoạn dùng lại **cùng lệnh**
với `--resume`. Resume từ chối khi code, môi trường, tham số hoặc dữ liệu
đổi. Không dùng chung output giữa CPU pilot và GPU run.

Đường NPZ cho sequence riêng: `train_x/test_x` dạng `(samples,frames,d)`
hoặc `(samples,frames)`; `train_labels/test_labels` dạng 1D. Độ dài biến
thiên dùng `train_values/test_values` dạng `(total_frames,d)` cùng
`train_offsets/test_offsets` là offsets integer tăng từ 0 đến total frames.
Không dùng object arrays hoặc pickle.

# Nhóm 1 GPU: đánh giá console log do người dùng cung cấp

Ghi nhận ngày 06/10/2026. Nguồn: console log và convergence CSV người dùng gửi
trong hội thoại. CSV server được lưu riêng tại
[gpu_convergence_user_supplied.csv](gpu_convergence_user_supplied.csv).
Chưa nhận summary GPU hoặc dòng tổng kết cuối runner. Các JSON và NPZ hiện
có trong thư mục report này vẫn là **CPU run**, không phải GPU.

## Những gì log xác nhận

Workload: TRAIN-only, 14 query × 16 gallery, `cuda:0`, hai profile default/tuned.
Mỗi profile có 6 synthetic case và 4 cặp TRAIN, ở 20/200 vòng.

| Profile | Parity checks đạt | Coupling relative L2 lớn nhất trong log | Case tại cực đại |
|---|---:|---:|---|
| default | 20/20 | 3.26e-6 | train_pair_2, 200 vòng |
| tuned | 20/20 | 4.14e-5 | synthetic_37_79_65, 200 vòng |

`passed` của từng case kiểm tra đồng thời score Eq.19, spatial cost,
coupling P và P@[1,Y] so với oracle affine FP64 ở cùng số vòng. Log chỉ in
lỗi coupling, vì vậy không suy ra cực đại sai số score/apply từ số này.
Ngưỡng coupling/apply đã khai báo là 5e-3; cực đại quan sát nhỏ hơn ngưỡng đó.

Cả 12 score matrix (hai profile × sáu checkpoint 20/100/200/500/1000/2000)
đều in `NN agreement=100%`. Đây là Flash FP32 chọn cùng nearest neighbor với
oracle CPU FP64 **ở cùng checkpoint** trên 14 query TRAIN.

## Diễn giải và giới hạn

* Phần parity trên 40 case đã đạt theo tiêu chí khai báo. Kết quả hỗ trợ rằng
  Flash tính đúng bài toán affine trong main PDF với precision FP32 IEEE
  trên các case đã kiểm tra, kể cả rectangular dài 1025×1537 và feature390.
* NN agreement=100% không phải ACC=100% theo ground-truth labels. Hai backend
  vẫn có thể cùng chọn một mẫu thuộc lớp sai.
* NN agreement của console so Flash tại t với FP64 tại t. Cột
  `nn_agreement_vs_last` trong CSV bổ sung phép so mỗi backend tại t với
  chính backend đó tại 2000 vòng: default trùng toàn bộ từ checkpoint20;
  tuned trùng13/14 ở20, và14/14 từ100. Đây là ổn định NN trên pilot TRAIN,
  không chứng minh toàn ranking/MAP hoặc TEST đã ổn định.
* CSV đã xác nhận marginal Flash thực tế. Ở200 vòng, tuned chỉ54/224 cặp đạt
  tau=1e-3, gần như đúng diễn biến của FP64; parity vẫn có thể đạt khi chưa
  hội tụ. Xem bảng hội tụ phía dưới.
* Console chưa in trạng thái numerical parity toàn ma trận score/spatial.
  Trong runner, NN agreement được in độc lập với boolean `passed` của matrix
  parity; 100% NN agreement không bảo đảm mọi score đạt ngưỡng số học.
* Chưa có căn cứ quy khoảng cách MAP/ACC giữa FlashOPW và OPW/TLp cho lỗi
  kernel; cũng chưa kiểm tra Taylor so với inverse exact. Đây là bài toán
  đối chứng khác của nhóm 2. Log này không đo tốc độ hoặc tái hiện journal.

## Hội tụ GPU đã được xác nhận từ CSV

| Profile | Vòng | Max marginal L1 GPU | Cặp đạt ≤1e-3 | Cặp đạt ≤1e-4 | Max score drift GPU so với 2000 |
|---|---:|---:|---:|---:|---:|
| default | 200 | 3.90e-4 | 224/224 | 209/224 | 9.42e-6 |
| default | 500 | 8.52e-5 | 224/224 | 224/224 | 2.09e-6 |
| default | 2000 | 1.92e-5 | 224/224 | 224/224 | 0, theo định nghĩa |
| tuned | 200 | 1.28e-2 | 54/224 | 1/224 | 1.81e-3 |
| tuned | 500 | 1.89e-3 | 218/224 | 69/224 | 7.61e-5 |
| tuned | 1000 | 2.69e-4 | 224/224 | 221/224 | 8.58e-6 |
| tuned | 2000 | 7.32e-5 | 224/224 | 224/224 | 0, theo định nghĩa |

* Checkpoint đầu tiên toàn bộ224 cặp đạt tau=1e-3: default200, tuned1000;
  với tau=1e-4: default500, tuned2000. Đây là kết luận trên pilot hiện tại,
  không phải số vòng tối thiểu chính xác hay bảo đảm cho mọi dataset/cặp dài.
* Các residual và score drift GPU gần FP64. Riêng tuned ở500 vòng, ngưỡng
  1e-4 được72/224 cặp FP64 đạt so với69/224 GPU. Sự khác biệt quanh ngưỡng
  nhỏ cần được ghi nhận; không thay residual GPU bằng giá trị reference.
* Median marginal GPU tại2000 là5.67e-7(default),6.55e-6(tuned), còn FP64
  tương ứng4.46e-16,1.13e-9. Diễn biến phù hợp với giới hạn precision FP32
  ở sai số nhỏ; chưa có sweep precision để xác định riêng từng nguyên nhân.
  Cả hai vẫn đạt ngưỡng1e-4 trên toàn bộ cặp.
* Không cần sửa kernel hoặc tự động thay selection chỉ vì tuned200 chưa
  hội tụ. Khi nhóm2 so prior/score, dùng cùng tiêu chí hội tụ đã khai báo;
  để benchmark nhóm5, báo cáo riêng fixed iterations và time-to-residual.
  Sweep hiện tại không đo thời gian dừng tại ngưỡng.
* Chưa quy thay đổi MAP/ACC cho thiếu iterations: NN tuned đã ổn định từ100
  trên14 query, nhưng cần full ranking và dữ liệu TEST để xét ảnh hưởng MAP.

## Phần cần bổ sung để đóng audit nhóm 1

Đã nhận và lưu convergence CSV; phần parity case và hội tụ đủ để rút kết luận
pilot ở trên. Còn `summary.json` từ `outputs/opw_group1_gpu_v1` để xác nhận
tổng `parity_failures`, numerical parity toàn matrix và full ranking agreement.
`parity.json`, `environment.json` và NPZ bổ sung provenance nếu gửi cả folder.
Đoạn người dùng gửi bắt đầu bằng `}` trước CSV, không chứa nội dung run_state;
không diễn giải dấu này thành trạng thái completed hoặc failed.

Lệnh đọc nhanh trên server:

```bash
cat outputs/opw_group1_gpu_v1/run_state.json
cat outputs/opw_group1_gpu_v1/convergence.csv
```

Không cần chạy lại nếu artifact đã hoàn tất. Gửi toàn folder để giữ source/data
fingerprints và score matrices phục vụ audit. Chỉ chạy `--resume` khi run còn
gián đoạn và code/settings/môi trường giữ nguyên.

# Nhóm 1 GPU: đánh giá console log do người dùng cung cấp

Ghi nhận ngày 06/10/2026. Nguồn: console log người dùng gửi trong hội thoại,
chưa nhận artifact GPU hoặc dòng tổng kết cuối của runner. Các artifact JSON,
CSV, NPZ hiện có trong thư mục report này vẫn là **CPU run**, không phải GPU.

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
* NN agreement ở từng checkpoint cũng không phải bằng chứng NN không đổi khi
  tăng iterations: runner so Flash tại t với FP64 tại t. Muốn kiểm tra sự ổn
  định qua iterations, đọc `versus_last_checkpoint` trong convergence output.
* Parity không chứng minh solver đã hội tụ. Oracle và Flash có thể cùng khớp
  một coupling còn sai marginal. Console không in marginal residual GPU.
  CPU run trước đó cho thấy tuned ở 200 vòng chỉ 54/224 cặp đạt tau=1e-3;
  cần residual **GPU thực tế** để xác nhận kết luận tương ứng trên Flash.
* Console chưa in trạng thái numerical parity toàn ma trận score/spatial.
  Trong runner, NN agreement được in độc lập với boolean `passed` của matrix
  parity; 100% NN agreement không bảo đảm mọi score đạt ngưỡng số học.
* Chưa có căn cứ quy khoảng cách MAP/ACC giữa FlashOPW và OPW/TLp cho lỗi
  kernel; cũng chưa kiểm tra Taylor so với inverse exact. Đây là bài toán
  đối chứng khác của nhóm 2. Log này không đo tốc độ hoặc tái hiện journal.

## Phần cần bổ sung để đóng nhóm 1

Nhận `summary.json`, `convergence.csv`, `parity.json`, `environment.json` từ
`outputs/opw_group1_gpu_v1`. Kiểm tra tổng `parity_failures`, matrix parity,
full ranking agreement, marginal và score drift ở từng checkpoint. Console
hiện dừng sau dòng NN tại 2000 vòng, chưa có dòng `Results: ...` cuối runner;
không coi đây là bằng chứng tổng run đã hoàn tất thành công.

Lệnh đọc nhanh trên server:

```bash
cat outputs/opw_group1_gpu_v1/run_state.json
cat outputs/opw_group1_gpu_v1/convergence.csv
```

Không cần chạy lại nếu artifact đã hoàn tất. Gửi toàn folder để giữ source/data
fingerprints và score matrices phục vụ audit. Chỉ chạy `--resume` khi run còn
gián đoạn và code/settings/môi trường giữ nguyên.

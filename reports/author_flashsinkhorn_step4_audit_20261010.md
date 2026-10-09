# Audit bước 4 FlashSinkhorn author — 10/10/2026

Lượt GPU chạy đủ **111/111 case**, không runtime error hoặc case thiếu;
**62 failed, 28 passed_checks, 21 passed_checks_with_coverage_gaps**.
Giữ nguyên trạng thái GPU `failed`. Chưa chuyển sang benchmark hiệu năng
bước 5 hoặc gọi đây là full correctness pass.

## Artifact và phạm vi audit

- Upload commit `bbf200c`, code chạy `ad3f12a`.
- ZIP `author_flashsinkhorn_step4_20261009_212437_3232203.zip`, 48598499 byte.
- SHA-256 `75118a0a5395c10557dff19e5165331161a3a3cb65049f221c78e2a5ee600170`.
- 341 thành viên gồm 340 dữ liệu và manifest; mọi checksum khớp, không tên trùng.
- Source trước/sau khớp 131 file tại pin `bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1`.
  Profile RTX 5080 khớp inventory, legacy trước/sau và bốn helper validation
  khớp hash local. Không sửa hai cây nguồn tác giả hoặc source solver cũ.
- Môi trường 111 case thống nhất: Python 3.14.8, Torch 2.11.0+cu128,
  Triton 3.6.0, CUDA 12.8, NumPy 2.4.6, RTX 5080; GeomLoss parent 0.3.1.

Audit toàn phần ban đầu dừng ở hash vector `direction`: CPU `torch.randn`
trên Windows không tái tạo đúng byte đã sinh trên Linux. **Cả 111 case có
x/y/a/b khớp từng byte**; chỉ direction khác. Bộ công cụ ban đầu đã dựa quá
nhiều vào giả định CPU RNG portable và không lưu vector thực tế.

Đã bổ sung chế độ **partial_audit** tường minh, không bỏ kiểm tra hash input
x/y/a/b. Tính lại đầy đủ plan, apply, gradient, các reference forward FP64,
residual và chênh lệch HVP từ hai output đã lưu. Kết quả partial audit bao
phủ 111 case, không lỗi nhất quán dữ liệu. **Chưa tính lại hệ KKT/direct-HVP
với đúng direction server**; 112 assertion direct-HVP của 56 case dense chỉ
được kiểm tra accounting, không được ghi là đã xác minh số học. HVP lớn cũng
chưa có direction xác thực; parity hai output được tính lại từ arrays lưu.

Vector không dùng trong 38 case forward nên hash direction không ảnh hưởng
audit phép toán của nhóm đó. 73 case có HVP cần bổ sung direction. Script
`export_author_step4_directions.py` tái sinh trên host gốc và chỉ xuất khi
**toàn bộ input hash khớp lượt GPU**, chia ZIP tối đa 48 MiB dữ liệu mỗi part.
Script chỉ chạy CPU, không chạy lại 111 phép đo GPU. Auditor nhận tất cả part
qua `--directions`, kiểm tra checksum, binding với SHA-256 ZIP gốc, dtype,
shape và hash direction từng case trước khi dùng.

ZIP gốc và dữ liệu audit/diagnosis đã sao lưu tại
`outputs/author_flashsinkhorn_diagnosis_bbf200c/` (ignored). Gỡ ZIP chuyển giao
khỏi main; lịch sử Git vẫn giữ commit upload. Báo cáo TeX đang chỉnh giữ nguyên.

## Kết quả theo nhóm

| Nhóm | Case | Failed | Passed | Passed với gap |
|---|---:|---:|---:|---:|
| Dense | 56 | 20 | 28 | 8 |
| Benchmark forward TF32 | 38 | 38 | 0 | 0 |
| Benchmark HVP IEEE | 17 | 4 | 0 | 13 |

Tất cả **111 phép so plan bản tác giả với bản cũ đã khớp lịch** đều đạt ngưỡng
đã khai báo; max relative L2 hàng chọn `4.63175024e-4`. Điều này không biến
các check so FP64 bị lỗi thành pass. Native symmetric và symmetric đã khớp
lịch được lưu riêng; không được so cùng tên n_iters rồi giả định cùng cập nhật.

35/56 dense case được xác nhận marginal/reference trong budget 500; số còn
lại chưa đạt tiêu chí đó. 73/73 cặp diagnostics HVP ghi CG converged theo
ngưỡng ban đầu, nhưng vẫn có HVP sai: cờ CG chỉ chứng nhận hệ tuyến tính đang
được giải, không chứng nhận các kernel còn lại hoặc độ chính xác toàn output.

## TF32: sai số cost được khuếch đại trong exponential

38/38 forward benchmark TF32 lỗi so operator FP64, dù các operator hai GPU
backend gần nhau. Max relative L2 author/legacy matrix apply chỉ khoảng
`5.89e-4`; chúng có thể cùng lệch nhiều so cost FP64 của input gốc.

Đã thử mô hình CPU cắt 13 bit fraction thấp của tọa độ FP32 trong dot product,
giữ norm IEEE, rồi dựng P với potentials lưu. Đây là **mô hình chẩn đoán**,
không phải CUDA rerun hoặc tiêu chí pass mới. Sai số dot đi vào exponent với
hệ số 2*cost_scale/epsilon; với tọa độ U[0,1], dimension cao và epsilon 0.1,
chênh lệch có thể lớn dù từng tọa độ chỉ mất ít bit.

| Case | Shape | Sai số mass so FP64 | Sai số mass so mô hình truncation |
|---|---|---:|---:|
| 001 | 17x23x1, eps=.01 | .0571040 | 1.94e-6 |
| 013 | 17x23x16, eps=.01 | .0196265 | 2.05e-6 |
| 065 | 20000x20000x4 | .0153728 | 1.10e-6 |
| 073 | 20000x20000x64 | .216831 | 1.01e-4 |
| 081 | 20000x20000x1024 | .968988 | .0337943 |

Các số từ arrays và input khớp hash hỗ trợ mạnh giả thuyết precision của dot
là nguyên nhân chính. Mô hình chưa tái hiện tất cả reduction/rounding Triton;
không gọi .968988 là sai số thuật toán OT riêng của author. Cần control IEEE
trên cùng tensor để chốt mức chính xác cho workload benchmark. Không nới
ngưỡng 5e-3 của lượt này để gọi TF32 đạt.

## HVP: lỗi lọc cấu hình Mat5 ở source ghim

Trong `kernels/apply_ott.py`, `_mat5_prune_configs` lấy dimension bằng
`named_args.get("D", 64)` và bỏ qua `kwargs`. Lời gọi kernel truyền `D=d`
bằng keyword. [Triton 3.6.0 Autotuner.run/prune_configs](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/autotuner.py#L194)
đặt positional arguments vào `named_args`, truyền keyword riêng cho hàm
prune; `D` không nằm trong dictionary mà code Mat5 đang đọc.

Tái hiện nguyên hàm prune trên CPU, không import Triton, cho thấy:

- `prune(configs, {}, D=512)` giữ BLOCK_D **64,128,256**, đều không đủ dimension.
- Control `prune(configs, {"D":512})` giữ BLOCK_D **512,1024,2048**.

Mat5 dùng grid một chiều, không tile qua D; output cấp phát bằng `torch.empty`.
Cấu hình BLOCK_D<d bỏ lại các cột chưa ghi. HVP còn không chuyển option
`autotune` của hàm ngoài xuống lời gọi `mat5_sqeuclid`, nên `autotune=False`
của runner không tắt autotuning Mat5. Đây là hai vấn đề code đã xác định tại
pin với API Triton 3.6, không phải suy luận từ speedup hoặc thiếu VRAM.

Dữ liệu lưu phù hợp trực tiếp: case_106 (10000x10000x512, IEEE) có norm HVP
author `1038.1179`, legacy `.1046227`, relative L2 **9922.4924**. Chênh lệch
trong từng nhóm 64 chiều đầu tới chiều 255 khoảng `1.37e-4`–`1.58e-4`; các
nhóm chiều 256–511 có norm chênh lệch **505–530**. Hai case IEEE d=512 và
hai case IEEE d=1024 nhỏ cũng có ranh giới sai rõ từ chiều 256.

Chưa có selected-config telemetry từ server hoặc GPU control Mat5 manual;
cần lưu cả hai khi kiểm tra biện pháp sửa. Không sửa bản tác giả readonly
hoặc âm thầm tính lại artifact thành pass. Các HVP nhỏ d=1 và các sai số nhỏ
hơn ở d=128/256 cần được phân tích tiếp bằng direction đúng và control CG
chặt hơn; chưa quy tất cả lỗi HVP cho cùng một nguyên nhân.

## Việc tiếp theo

1. Bổ sung đúng direction bằng exporter CPU trên server, audit lại direct-HVP
   từ archive gốc; không chạy lại toàn bộ 111 GPU case chỉ để chuyển input.
2. Control GPU có telemetry: Mat5 nguyên bản so manual block đủ D, lặp lại
   để phát hiện cột không ghi; IEEE so TF32 trên cùng input; CG chặt hơn cho
   các case nhỏ nhạy. Nếu dùng workaround, giữ riêng và ghi đúng patch/flags.
3. Chỉ chốt correctness và tiến sang bước 5 sau khi các lỗi này được xử lý
   hoặc ghi rõ giới hạn áp dụng. Coverage gap OTT-Hessian ở bước 3 vẫn còn.

Scripts mới được kiểm tra CPU; kết quả đó không thay cho chạy CUDA control.

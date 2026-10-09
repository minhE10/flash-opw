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

Audit ban đầu chỉ thực hiện một phần vì CPU `torch.randn` trên Windows không
tái tạo đúng byte direction đã sinh trên Linux. **Cả 111 case có x/y/a/b khớp
từng byte**. Đã khắc phục phần thiếu input bằng exporter CPU chạy trên host gốc;
exporter chỉ xuất khi toàn bộ input hash khớp lượt GPU, không chạy lại solver.

Đã nhận ba ZIP direction qua commit **`71e2990`**. Manifest từng part khớp
checksum và SHA-256 archive GPU gốc; đủ **73/73 direction HVP**, đúng dtype,
shape và hash input của từng case. Không còn assertion direct-HVP chưa audit.
38 case forward không dùng direction.

| Part | Byte | SHA-256 |
|---|---:|---|
| part_01.zip | 27993783 | `4eab5375c48637df3e81288fb792c55d9602c912cd1be36a9d7621c641ad2dff` |
| part_02.zip | 40309674 | `b7d3aae5a4850a0adede0204e23c86002cd5de04afc390dc6889be3eec0d07e3` |
| part_03.zip | 11855631 | `b55342799a4f4c3d03824516fcaedc0803afc2c8260791a5e8b43590625e3cc9` |

**Audit đầy đủ dữ liệu đã lưu hoàn tất cho 111/111 case, errors=[]**: tính lại
plan/apply/gradient, reference forward FP64, residual, parity HVP và hệ KKT
FP64 với direction đúng cho 56 case dense. Cả **112 assertion direct-HVP**
được tính lại, khớp sai số và trạng thái gốc; reference HVP lưu khớp reference
tính lại ở relative L2 <1e-8, residual hệ tuyến tính <=1e-9.
Trạng thái audit là `failed` vì xác nhận các lỗi số học gốc, không phải
`failed_audit` do dữ liệu thiếu hoặc không nhất quán. Không chạy lại CUDA;
HVP lớn vẫn chỉ kiểm tra các hàng lưu, chưa có direct-KKT toàn shape.

ZIP gốc sao lưu tại `outputs/author_flashsinkhorn_diagnosis_bbf200c/`; ba ZIP
direction và audit đầy đủ tại `outputs/author_flashsinkhorn_diagnosis_71e2990/`
(đều ignored). Bản sao đã kiểm tra SHA-256 trước khi gỡ ZIP chuyển giao khỏi
main; lịch sử Git vẫn giữ commit upload. Báo cáo TeX đang chỉnh giữ nguyên.

## HVP đối chiếu trực tiếp với FP64 bằng direction đúng

| Backend | Đạt ngưỡng 5e-4 | Không đạt | Max relative L2 |
|---|---:|---:|---:|
| Author | 44/56 | 12 | 24.6876982 |
| Legacy trên cùng plan và damping đúng đơn vị | 56/56 | 0 | 2.95629229e-5 |

12 case author không đạt là `case_000`–`case_003` (d=1), `case_032`–`case_035`
(d=512), `case_036`–`case_039` (d=1024), bao gồm hai lịch cập nhật và hai cấu
hình precision của forward. HVP của cả hai backend luôn chạy IEEE trong lượt
này. Plan được cấp chung cho phép so HVP; lỗi forward TF32 được kiểm tra riêng.

Các case IEEE tiêu biểu:

| Case | Shape | Author/direct | Legacy/direct |
|---|---|---:|---:|
| 000 | 17x23x1 | .00119192310 | 2.95629229e-5 |
| 002 | 17x23x1 | .00139204953 | 2.11191486e-5 |
| 032 | 17x23x512 | .619416368 | 2.99386083e-7 |
| 036 | 17x23x1024 | 24.6876944 | 2.46525560e-6 |

Với case 032, norm chênh lệch author/legacy ở 256 chiều đầu là `6.25e-7`,
ở phần còn lại là `6.78151`; case 036 tương ứng `5.43e-6` và `206.986`.
Direction đúng xác nhận ranh giới sai từ chiều 256. Thử giả định các cột Mat5
chưa ghi bằng zero rồi cộng bù Mat5 FP64 **không** giải thích được output lưu;
`torch.empty` không bảo đảm zero. Không dùng phép cộng bù đó để sửa kết quả.
Vẫn cần chạy control GPU có telemetry và manual block đủ D.

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
hoặc âm thầm tính lại artifact thành pass. Direction đúng đã xác nhận lỗi
author/direct ở d=1, dù legacy/direct đạt. Chỉ còn giả thuyết CG/FP32 cho nhóm
này; cờ CG hội tụ không bảo đảm ngưỡng HVP 5e-4. Các sai số nhỏ hơn ở d=128/256
của shape lớn chưa có đối chiếu direct-KKT, cần control GPU riêng. Chưa quy
tất cả lỗi HVP cho cùng một nguyên nhân.

## Việc tiếp theo

1. Đã hoàn tất bổ sung direction và audit direct-HVP từ archive gốc.
2. Tiếp theo: control GPU có telemetry, Mat5 nguyên bản so manual block đủ D, lặp lại
   để phát hiện cột không ghi; IEEE so TF32 trên cùng input; CG chặt hơn cho
   các case nhỏ nhạy. Nếu dùng workaround, giữ riêng và ghi đúng patch/flags.
3. Chỉ chốt correctness và tiến sang bước 5 sau khi các lỗi này được xử lý
   hoặc ghi rõ giới hạn áp dụng. Coverage gap OTT-Hessian ở bước 3 vẫn còn.

Scripts mới được kiểm tra CPU; kết quả đó không thay cho chạy CUDA control.

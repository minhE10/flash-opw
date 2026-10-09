# Audit FlashSinkhorn author: bước 1–2, 09/10/2026

Artifact nhận qua commit `dfb4166`:
`author_flashsinkhorn_steps12_20261009_193620_3223378.zip`.
Trạng thái tổng vẫn là **`failed_or_incomplete`**: bước 1 có lỗi baseline và
thiếu API; bước 2 đạt các tiêu chí chẩn đoán đã đặt trước.

## Kiểm tra artifact và nguồn

- SHA-256 ZIP: `19de1dd6564a5031260b5f99ff9a68a265ba9d1ef0ae3099d6b16f5b722c3f95`.
- 17 thành viên dữ liệu khớp manifest; không có tên trùng hoặc file ngoài manifest.
- Metadata nguồn trước/sau kiểm tra xác nhận 131 file tại pin
  `bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1`. Hai inventory profile RTX 5080
  khớp từng hash với profile được tạo local: chỉ đổi launch matrix apply.
- Metadata dependency trước/sau khớp 22 hash OTT-Hessian tại pin
  `7eb189fe39982f587da935044480655b65939637`.
- JUnit tính lại: **3 test, 1 failed, 2 skipped, 0 passed**; khớp summary.
- Input CUDA lưu trong JSON khớp SHA-256
  `7e02143f7b62436b4b32919f3165b71cae660f8964a8b1b3047b06beaedde54e`.

ZIP đã được sao lưu và kiểm tra checksum tại
`outputs/author_flashsinkhorn_diagnosis_dfb4166/`. Thư mục ignored này giữ ZIP,
`audit_steps12.py`, `audit.json` và input; ZIP chuyển giao được gỡ khỏi main sau
audit. Git history vẫn giữ commit upload. Không sửa source/test tác giả.

## Bước 1: xác định nguyên nhân HVP không đạt

Test KeOps lỗi **trước phép so sánh HVP**, trong dependency OTT-Hessian:

```text
ValueError: Incompatible values for attribute ni: 16 and 256.
torch_sinkhorn_hessian.py:950: Mat1 = apply_axis1(y.t())
torch_sinkhorn_hessian.py:754: result = (K_ij * arr_i).sum(dim=0)
```

Fixture có 256 điểm, dimension 16. `y.t()` có shape `(16, 256)`; baseline
dùng `LazyTensor(arr[:, None, :])`, tạo shape `(16, 1, 256)` và coi 16 là
chiều điểm i. Kernel có chiều i bằng 256, nên KeOps từ chối phép nhân.
Đây là lỗi bố trí tensor trong đường apply của baseline public được ghim.
Traceback không phải lỗi OOM, shared memory hay assertion sai số FlashSinkhorn.
Nó cũng chưa cung cấp kết quả parity HVP thành công cho fixture này.

Observer xác nhận thực sự dùng KeOps, không fallback dense. Giữ nguyên
`tau2=1e-5`, cap CG 50, `cg_rtol=cg_atol=1e-6`, không preconditioner.
Baseline lỗi trước khi có CG diagnostics để đánh giá hội tụ.

Hai test còn lại skip vì import API JAX yêu cầu `HessianALineax` không có
trong pin. Khảo sát 34 commit reachable từ các branch đã fetch của clone
không shallow cũng không thấy tên API này. Kết luận chỉ áp dụng cho nguồn
public đã khảo sát. Không alias API hoặc sửa dependency để gọi là test gốc đạt.

Bước 1 đã làm rõ nguyên nhân từng nonpass, **chưa hoàn tất đối chứng HVP ngoài**.
Nếu sửa baseline trong bản dẫn xuất, cần kiểm chứng riêng phép apply và ghi rõ
patch; kết quả đó phải phân biệt với test trên dependency public nguyên bản.

## Bước 2: early stopping và hội tụ

Đúng fixture gốc: CUDA RNG seed 0, x/y FP16, weights FP32, shape
`128 × 128 × 32`, epsilon 0.1, threshold `1e-3`, check every 5; giữ các flags
TF32, exp2 và extrapolation. Khối matrix apply dùng profile RTX đã mô tả.
Không đổi tolerance/assertion upstream hoặc chuẩn hóa lại weights đã làm tròn.

Reference log-domain alternating CPU FP64 trên input CUDA đã lưu đạt
**52.500 vòng**, marginal L1 **`9.646659225477899e-7 <= 1e-6`**, cap 64.000.
Đã chạy lại phép solve này bằng Torch CPU FP64 tại local trên cùng input:
số vòng và residual khớp dữ liệu server trong ngưỡng kiểm tra
`rel_tol=1e-5, abs_tol=1e-12`; không dùng kết quả local làm timing GPU.

| Budget | Số cập nhật thực, threshold 1e-3 | Marginal L1 | Plan relative L2 so FP64 hội tụ | Xác nhận dừng theo potentials |
|---:|---:|---:|---:|:---:|
| 200 | 202 | 0.0792973330 | 0.396386549 | Không |
| 1.000 | 1.002 | 0.0106546413 | 0.102831176 | Không |
| 4.000 | 4.002 | 0.0005148774 | 0.013079230 | Không |
| 16.000 | 11.382 | 0.0001147325 | 0.002973481 | Có |

Hai cập nhật thêm đến từ khởi tạo và extrapolation cuối. Ba budget đầu cho
cùng output với control fixed-budget tương ứng. Tại budget 200, plan GPU
so reference symmetric FP64 **cùng lịch cập nhật** có relative L2
`2.4456241566558475e-5`: hỗ trợ kết luận cảnh báo ở fixture này do budget
chưa đủ, không phải bằng chứng kernel tính sai nghiệm fixed-budget.

Tại 11.382 cập nhật, đồng thời đạt marginal `<=1e-3`, plan relative L2
`<=5e-3`, xác nhận potential stop và không có warning. Control fixed 16.000
có 16.002 cập nhật, marginal `1.9667808595639352e-5`, plan relative L2
`1.1746737189851924e-4`. Budget 4.000 chỉ đạt marginal; chưa đạt tiêu chí
plan và chưa xác nhận potential stop.

Bước 2 **đạt trên fixture này**, trạng thái `marginal_and_reference_confirmed`.
Các ngưỡng trên là tiêu chí chẩn đoán riêng, không thay ngưỡng test tác giả.
Kết quả không chứng minh hội tụ ở mọi shape/epsilon, cũng không đo tốc độ.

## Giới hạn audit và bước tiếp theo

Audit kiểm tra checksum, inventory, metadata, JUnit và tính nhất quán các
tiêu chí. Input lưu cho phép chạy lại reference CPU FP64 tại local. Artifact
không lưu GPU potentials/plan; vì vậy các residual và sai số GPU trong bảng
là số đo đã lưu, không phải tính lại độc lập từ GPU output ở local.
Hash metadata không thay việc chạy lại CUDA hoặc xác thực danh tính server.

Giữ nguyên hai cây nguồn tác giả. Bước 3 là chạy lại đầy đủ 20 file trong
một môi trường thống nhất, ghi rõ coverage gap OTT-Hessian; không tính test
lỗi/skip thành pass. Bước 4–6 trong
[kế hoạch kiểm tra](../docs/flash_sinkhorn_author_validation_plan.md) vẫn chưa thực hiện.

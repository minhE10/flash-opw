# Audit FlashSinkhorn author: lượt full bước 3, 09/10/2026

Đã chạy đủ **20 file** trong một lượt, tổng **404 test: 401 passed, 1 failed,
2 skipped, 0 errors**. Cả 19 file ngoài `test_hvp_parity.py` đạt các assertion
upstream; file HVP có một lỗi baseline KeOps và hai skip API JAX đã biết từ
bước 1. Trạng thái tổng giữ nguyên **`failed`**, chưa gọi là full pass.

## Artifact và kiểm tra nguồn

- Upload commit: `9bd2535`; project chạy: `9ee8a8db114bc9f0ebe1129891eba38e3412b655`.
- ZIP: `author_flashsinkhorn_step3_20261009_201015_3069439.zip`, 376190 byte.
- SHA-256: `18f889094aa0868a8829c42478f6184451ee8e2c5e94a11da19993de14af6ceb`.
- 151 thành viên dữ liệu khớp checksum manifest; không có tên trùng hoặc
  thành viên ngoài manifest. JUnit từng file khớp số đếm và summary full.
- Metadata nguồn trước/sau từng file và toàn lượt xác nhận 131 file tại pin
  `bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1`, không có lỗi integrity.
- Inventory hash của cả 20 profile test và profile independent khớp bản
  dẫn xuất RTX 5080 tạo local. Chỉ thay launch matrix apply; assertion và
  tolerance tác giả giữ nguyên. Trạng thái kiểm tra profile sau chạy đều đạt.
- Dependency OTT-Hessian trước/sau toàn lượt và trước/sau file HVP khớp
  22 hash tại pin `7eb189fe39982f587da935044480655b65939637`.
- Audit tự động trả `status=failed, errors=[]`: kết quả test có lỗi nhưng
  bằng chứng nhất quán, không phải `failed_audit`.

Đã sao lưu ZIP, summary, checksum, numerical diagnostics, CG và kết quả audit
tại `outputs/author_flashsinkhorn_diagnosis_9bd2535/` (ignored), kiểm tra lại
SHA-256 bản sao trước khi gỡ ZIP chuyển giao khỏi main. Git history vẫn giữ
commit upload. Không sửa hai cây nguồn tác giả hoặc báo cáo TeX đang chỉnh.

## Môi trường thống nhất

Cả 20 preflight record có cùng phiên bản Python, Torch, Triton, CUDA, GPU,
compute capability, FlashSinkhorn và dependency. GeomLoss trước/sau toàn lượt
đều 0.3.1. GPU RTX 5080, compute capability 12.0; Python 3.14.8,
Torch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, pytest 9.1.1, NumPy 2.4.6,
GeomLoss 0.3.1, KeOps 2.3, JAX/JAXlib 0.8.2, OTT-JAX 0.5.1, Lineax 0.0.8.
Mỗi file dùng process mới, cùng interpreter và cấu hình validation.

## Kết quả từng file

| File | Passed | Failed | Skipped |
|---|---:|---:|---:|
| test_apply_plan_flashstyle.py | 27 | 0 | 0 |
| test_autograd_semantics.py | 21 | 0 | 0 |
| test_c_transform.py | 17 | 0 | 0 |
| test_flashstyle_parity.py | 21 | 0 | 0 |
| test_geomloss_sinkhorn_triton.py | 10 | 0 | 0 |
| test_geomloss_vs_triton.py | 4 | 0 | 0 |
| test_half_cost.py | 7 | 0 | 0 |
| test_hvp_parity.py | 0 | 1 | 2 |
| test_hvp_sqeuclid.py | 4 | 0 | 0 |
| test_identities.py | 10 | 0 | 0 |
| test_multiscale_kernels.py | 40 | 0 | 0 |
| test_multiscale_solver.py | 49 | 0 | 0 |
| test_ott_vs_triton.py | 16 | 0 | 0 |
| test_pad_to_multiple.py | 31 | 0 | 0 |
| test_samples_loss_api.py | 7 | 0 | 0 |
| test_samples_loss_tf32.py | 53 | 0 | 0 |
| test_semi_unbalanced_forward.py | 6 | 0 | 0 |
| test_sinkhorn_triton.py | 10 | 0 | 0 |
| test_unbalanced_reference.py | 46 | 0 | 0 |
| test_unbalanced_sinkhorn.py | 22 | 0 | 0 |
| **Tổng** | **401** | **1** | **2** |

Test HVP KeOps tiếp tục lỗi `Incompatible values for attribute ni: 16 and
256` trong layout tensor của baseline OTT-Hessian, trước phép so HVP.
Observer xác nhận dùng KeOps, không fallback; giữ `tau2=1e-5`, cap CG 50,
rtol/atol `1e-6`, không preconditioner. Hai test JAX skip vì import API
`HessianALineax` còn thiếu. Xem phân tích nguồn trong
[audit bước 1–2](author_flashsinkhorn_steps12_audit_20261009.md).
Không coi ba nonpass này là pass hoặc bằng chứng kernel FlashSinkhorn sai.

## CG và đối chứng FP64 độc lập

Fixture double-backward của API được kiểm tra riêng với cap CG 256.
Cả đường autograd và đường gọi reference đều hội tụ sau **149 bước**, residual
**`5.446255499919062e-7 <= 1e-6`**, output hữu hạn. Giữ damping và tolerance;
assertion parity gốc không đổi. Reference trong fixture này dùng cùng backend,
khác với đối chứng FP64 độc lập bên dưới.

**12/12 case FP64 độc lập đạt**, gồm ba shape `(17,23,3)`, `(32,24,16)`,
`(37,29,33)`, hai schedule và hai cost scale 0.5/1.0; IEEE FP32 trên GPU.
Kiểm tra cả hai chiều apply; mọi CG trong nhóm này xác nhận hội tụ.

| Metric | Giá trị lớn nhất |
|---|---:|
| Reference marginal L1 | 4.8428773941e-8 |
| GPU marginal L1 | 1.1646306833e-7 |
| Plan relative L2 | 2.1062109393e-7 |
| HVP relative L2 | 4.8880040973e-7 |
| Direct system relative residual | 3.9627312345e-16 |

Không cộng 12 case độc lập vào 404 test pytest. Phạm vi nhỏ này chưa thay
kiểm tra HVP bên ngoài còn thiếu hoặc correctness trên toàn bộ benchmark.

## Warnings và kết luận phạm vi

Warnings vẫn được lưu: deprecation từ API cũ/JAX dependencies; cảnh báo
potential-change early stopping tại budget 200, các fixture padding dùng
budget ngắn và một số fixture unbalanced/epsilon schedule. Không ghi nhận
cảnh báo CG không hội tụ trong các pytest log của lượt này.

Fixture balanced budget 200 đã được chẩn đoán riêng ở bước 2: cần thêm
cập nhật để đạt các tiêu chí marginal/reference trên input đó. Kết quả này
không chứng nhận hội tụ cho mọi lời gọi padding, unbalanced hoặc epsilon
schedule có warning. Test pass chỉ xác nhận các assertion nó thực sự kiểm tra.

**Bước 3 đã hoàn tất lượt chạy và audit; còn đối chứng OTT-Hessian chưa đạt.**
Đây là bằng chứng correctness trong phạm vi validation với profile RTX,
không phải benchmark hiệu năng hoặc chứng nhận mọi cấu hình đều hội tụ.
Audit kiểm tra dữ liệu đã lưu, không chạy lại CUDA và không xác thực danh
tính server. Bước 4–6 của
[kế hoạch](../docs/flash_sinkhorn_author_validation_plan.md) chưa thực hiện.

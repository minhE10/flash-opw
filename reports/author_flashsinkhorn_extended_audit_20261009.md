# Audit FlashSinkhorn author mở rộng — 09/10/2026

Đã kiểm tra artifact GPU `author_flashsinkhorn_extended_20261009_154429_3212761.zip`
nhận qua commit `e34c467`. Auditor tính lại checksum, số test trong JUnit,
inventory 20 file, metadata hash profile RTX 5080, CG và metrics FP64 độc lập.
Không có lỗi integrity trong bằng chứng đã lưu (`errors: []`); trạng thái bộ
test vẫn là **failed**, gồm **374 passed, 1 failed, 2 errors, 3 skipped**.
Đây là audit artifact, không phải chạy lại GPU tại máy local.

Nguồn tác giả ghim tại `bb2bf5aeaf2eacbbf291b767e14d97a6702dc9e1`, version 0.4.1.
Cả 20 file đã được thử; không có file chưa chạy. Profile chỉ điều chỉnh launch
matrix apply cho RTX 5080; các test và tolerance tác giả giữ nguyên.

## Nguyên nhân và phần còn thiếu

| File | Kết quả chưa đạt | Nguyên nhân trong traceback |
|---|---|---|
| `test_geomloss_sinkhorn_triton.py` | 1 failed | `ModuleNotFoundError: geomloss._legacy` |
| `test_geomloss_vs_triton.py` | 1 collection error | Cùng lỗi import `_legacy.sinkhorn_samples` |
| `test_unbalanced_sinkhorn.py` | 1 collection error | Cùng lỗi import `_legacy.sinkhorn_samples` |
| `test_hvp_parity.py` | 3 skipped | Thiếu module OTT-Hessian bên ngoài: `torch_sinkhorn_hessian`, `SinkhornHessian` |

Môi trường dùng GeomLoss 0.2.6, trong khi dev extras của commit tác giả yêu cầu
`geomloss>=0.3`. Ba nonpass trên là lỗi dependency, chưa có assertion số học
thất bại trong những test đã chạy được. Không coi collection error là pass.

Wheel GeomLoss 0.3.1 đã được tải và kiểm tra ở local: có các import
`sinkhorn_tensorized`, `softmin_tensorized`, `dampening` dưới `geomloss._legacy`.
File `requirements-author-validation.txt` ghim version và SHA-256 wheel.
Chỉ cài package này vào venv author với `--no-deps`; lượt GPU sau cần chạy lại
cả năm file import GeomLoss bên ngoài. Chưa có kết quả GPU với dependency mới.

Ba skip OTT-Hessian vẫn là khoảng trống coverage riêng. Checkout public đã
kiểm tra tại `7eb189fe39982f587da935044480655b65939637` có module KeOps nhưng
`SinkhornHessian.py` không định nghĩa `HessianALineax` mà test JAX yêu cầu.
Không thay class/thuật toán chỉ để biến skip thành pass. Test OTT-JAX native
`test_ott_vs_triton.py` đã đạt 16/16 trong artifact hiện tại.

## Kiểm chứng độc lập đã đạt

Cả 12 case CPU FP64/GPU FP32 độc lập đạt trên ba shape, hai cost scale và hai
solver. Các case dùng weights không đều, kiểm tra plan, marginal, apply hai
trục và HVP đối chiếu hệ KKT giải trực tiếp FP64.

| Metric | Giá trị lớn nhất |
|---|---:|
| GPU marginal L1 | 1.1646306832986675e-7 |
| Plan relative L2 | 2.1062109392902477e-7 |
| HVP relative L2 | 4.888004097285891e-7 |

Tất cả CG của 12 case xác nhận hội tụ. Fixture API double backward cũng ghi
hai đường autograd/reference hội tụ sau 149 bước, residual
5.446255499919062e-7 <= 1e-6 tại cap 256, giữ nguyên damping/tolerance.
Các warnings early stopping ở những fixture khác vẫn được giữ lại; các số
trên không chứng minh mọi lời gọi solver trong full suite đã hội tụ.

## Lưu và dọn artifact

SHA-256 ZIP:
`517769b29e1ba0e92bac2bad55767cc8f02aad335147e10b076e745d5b6ad3e3`.
Bản local có cùng hash nằm trong thư mục ignored
`outputs/author_flashsinkhorn_diagnosis_e34c467/`; ZIP được gỡ khỏi nhánh main
ở commit `3499c3f` sau audit. File vẫn tồn tại trong lịch sử Git.
Theo yêu cầu người dùng, những file server cần xem tiếp sẽ được chuyển bằng
commit/push, pull để kiểm tra, giữ bản local rồi gỡ khỏi repo sau kiểm tra.

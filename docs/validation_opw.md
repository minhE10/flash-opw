# Kiểm chứng FlashOPW — 06/10/2026

Chuẩn công thức là `main (2).pdf`, bản 3 trang ngày 02/10/2026. SHA256:
`2b1327ebe67e33edbc0ca640e5223de9b16f667fc1bdb70c4b2e4b2f720dbe22`.
Journal cung cấp baseline và protocol; SHA256 `OWD_journal.pdf`:
`7a5ff792703b3153efdfeba4c263fbfecd548bf7a22620f69947433e2690e7fb`.
Xem [công thức và lệnh chạy](flash_opw.md).

## Kiểm tra implementation

Môi trường local: Windows, Python 3.14.3, Torch 2.11.0+cpu, SciPy 1.17.1,
Numba 0.65.1. Chưa chạy CUDA/RTX 5080 trong phiên này.

- Toàn bộ test suite sau tách package: **88 passed, 136 skipped**.
  Các skip do GPU hoặc dependency chưa có. Sau chỉnh preset số vòng và
  kiểm tra tham số rất nhỏ/lớn, chạy lại nhóm OPW/data/metrics/retrieval:
  **42 passed, 12 GPU tests skipped**.
- Cost mở rộng, thế f/g, coupling, loss main (19) và spatial score khớp
  recurrence dense độc lập; kiểm tra cả marginal không đều và hai schedule.
- Test affine/Taylor ghi rõ sai số so với inverse chính xác, không coi
  hai cost là bằng nhau. MAP và majority ACC được đối chiếu ví dụ tính tay.
- Sáu file của lõi FlashSinkhorn giữ nguyên nội dung sau chuẩn hóa CRLF/LF
  so với commit `9d3dbae`; chuyển từ `flashopw` sang `flashsinkhorn`.
- Hai script mới qua kiểm tra cú pháp Bash trong WSL; compile Python và
  `git diff --check` qua.
- Integration với dữ liệu nhỏ chạy đủ 12 metric ở preset FlashOPW 200 vòng;
  resume giữ nguyên CSV, đổi số vòng bị từ chối. Không dùng accuracy của
  dữ liệu nhỏ này làm bằng chứng hiệu năng phân loại.

12 test CUDA mới đối chiếu FlashOPW FP32 với dense FP64 ở d không gian
`1,63,64,65,390,1023`, hai schedule. Cần chạy trên server bằng
`bash scripts/validate_flash_opw.sh 1`; skip local không phải GPU pass.

## Pilot dữ liệu thật, 20 vòng

FacesUCR TRAIN/TEST chính thức, 200 train/2050 test, 14 class, 131 frame,
1 feature. Pilot chọn cân bằng 32 gallery, 16 query với seed 42/43;
không thay giá trị frame. Đây là run kiểm tra đầu tiên trước khi tăng
preset FlashOPW lên 200 vòng; output ghi rõ số vòng 20.

TRAIN SHA256: `dda6897df22db05960b00a5147f53346774848bf86ccb4d918fde201eab7ad0f`.
TEST SHA256: `042b5c265c2c88686abe7b544eb78888ba5ebfd787dc21d6368080dd0883456d`.
Không có metric lỗi; toàn bộ score matrices đều hữu hạn.

| Metric | MAP (%) | ACC@1 | ACC@3 | ACC@5 | ACC@7 | ACC@15 | ACC@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affine OPW Torch streaming, main (19) | 68.329 | 62.50 | 62.50 | 68.75 | 62.50 | 31.25 | 37.50 |
| Affine OPW dense, main (19) | 68.329 | 62.50 | 62.50 | 68.75 | 62.50 | 31.25 | 37.50 |
| DTW | 62.778 | 62.50 | 56.25 | 56.25 | 62.50 | 43.75 | 37.50 |
| lDTW | 62.778 | 62.50 | 56.25 | 56.25 | 62.50 | 43.75 | 37.50 |
| nDTW | 52.337 | 37.50 | 56.25 | 50.00 | 50.00 | 37.50 | 37.50 |
| Soft-DTW | 69.055 | 75.00 | 68.75 | 62.50 | 62.50 | 37.50 | 37.50 |
| OT | 56.937 | 56.25 | 56.25 | 62.50 | 68.75 | 43.75 | 31.25 |
| Sinkhorn | 57.363 | 56.25 | 56.25 | 62.50 | 75.00 | 43.75 | 31.25 |
| TLp | 82.482 | 81.25 | 81.25 | 81.25 | 75.00 | 43.75 | 31.25 |
| OPW-KL | 84.034 | 93.75 | 93.75 | 87.50 | 75.00 | 56.25 | 31.25 |
| TCOT | 64.225 | 62.50 | 68.75 | 68.75 | 62.50 | 43.75 | 31.25 |
| OPW journal | 84.034 | 93.75 | 93.75 | 87.50 | 75.00 | 56.25 | 31.25 |

Hai implementation affine FP64 cho cùng ranking 1-NN ở mọi query;
max score error `3.33e-16`, relative L2 `2.30e-16` trên đủ 512 cặp.
Đây là Torch streaming CPU, **chưa phải kết quả FlashOPW CUDA**.
OPW và OPW-KL có cùng MAP/ACC trên subset này không có nghĩa chúng cho
cùng coupling hoặc score.

Với 8 cặp diagnostics, sau 20 vòng row L1 residual từ khoảng 0.0086 đến
0.0365; column L1 khoảng `1e-15`. 20 vòng chưa hội tụ. Trên ba cặp kiểm tra
thêm, 200 vòng giảm row residual xuống `6.39e-6`–`4.82e-5`; API dense với
`n_iters=1000,tol=1e-8` dừng ở 380–480 vòng, cả hai marginal dưới `1e-8`.
Đó là lý do tách số vòng FlashOPW theo main khỏi preset 20 vòng journal.

Chạy thêm affine dense **200 vòng trên cùng đủ 512 cặp**, cùng subset và
tham số: MAP **68.449%**, ACC ở cả sáu k giữ nguyên so với 20 vòng. Run ở
`outputs/opw_main200_cpu_dense_20261006/` đã hoàn tất qua checkpoint/resume.
Không dùng wall time của run bị gián đoạn này để so sánh tốc độ. Kết quả
200 vòng là đối chứng FP64 cho preset server; chưa có ma trận Flash CUDA
200 vòng để kết luận parity trên server.

Output local đầy đủ ở `outputs/opw_knn_cpu_pilot_20261006/`: CSV/JSON,
distance matrices, AP/predictions, parity, diagnostics, provenance và hình
MAP/ACC. `environment.json` ghi commit gốc và source hashes của code chưa
commit tại lúc chạy. Output và dữ liệu không được đưa lên Git.

Chưa kết luận tái hiện thành công các bảng journal hoặc FlashOPW nhanh
hơn baseline: subset nhỏ, tham số cố định, ground cost bình phương theo
main và backend CPU/CUDA khác nhau. Bước tiếp theo là validation CUDA rồi
pilot cùng split/seed trên server, kiểm tra parity và marginal trước khi
mở rộng số query hoặc chạy toàn bộ split.

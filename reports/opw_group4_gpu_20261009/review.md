# Nhóm 4: kết quả official TEST FacesUCR

Đọc và kiểm tra ngày **09/10/2026** từ commit kết quả **8cfb172**.

**Kết quả chính:** FlashOPW selected đạt **ACC@1 92,244% / MAP 69,946%**. Nó cao hơn OPW journal selected, gần TLp về ACC nhưng thấp hơn TLp về MAP. OPW journal preset lại có điểm cao hơn FlashOPW selected; dữ liệu không hỗ trợ kết luận FlashOPW tốt hơn mọi cấu hình OPW hoặc mọi baseline.

## Phạm vi và kiểm tra

- Gallery: toàn bộ 200 TRAIN; query: toàn bộ 2050 TEST; FacesUCR, 14 lớp, độ dài 131, một feature.
- 17 cấu hình duy nhất, 22 method/profile evaluations, 8.721 chunks, 6.970.000 cặp được tính.
- MAP toàn gallery; ACC@k tại 1/3/5/7/15/30; k=1 là tiêu chí chính.
- Tham số đã chọn trên ba TRAIN holdout được giữ nguyên; tolerance 0,001, check mỗi 50 vòng, cap 4000.
- Main dùng literal Eq.19; OPW journal giữ inverse exact, prior perpendicular và score `<P,D>`.
- Server audit báo **passed**, bao gồm chunk hashes, complete matrices, per-query AP/predictions và thống kê.
- Local readback **passed**: 13 file trong manifest (khôi phục xuống dòng LF/CRLF trước kiểm tra SHA256), 21 source hashes, fingerprint dữ liệu TRAIN/TEST, tham số/policy/migration, 132 dòng metric/k và 20 paired comparisons.
- Đã tính lại MAP từ AP lưu sẵn, ACC từ predictions với nhãn TEST chính thức; tính lại bootstrap CI, exact McNemar và Holm. Không có chunks/matrices trong gói upload nên không tính lại AP từ ranking, residual từ coupling hay chạy lại solver trên GPU ở local.
- Full bản gốc đã lưu trong `outputs/opw_group4_gpu_import_20261009/original_reports/`. Script kiểm tra: `audit_received.py`; kết quả: `audit_received.json`.

Lần chạy này đóng băng tham số TRAIN trước full evaluation. Tuy nhiên, các pilot TEST đã được xem trong quá trình phát triển trước đây; không gọi toàn bộ quá trình phát triển là blind TEST.

## Bảng selected

| Metric | ACC@1 % | MAP % | Cặp marginal L1 > 0,001 |
|---|---:|---:|---:|
| FlashOPW main | 92,244 | 69,946 | 5 |
| OPW journal | 89,707 | 66,576 | 0 |
| TLp | 92,341 | **70,432** | 0 |
| OPW-KL | 89,659 | 66,503 | 0 |
| Sinkhorn | 60,732 | 42,307 | 0 |
| TCOT | 63,951 | 45,470 | 0 |
| DTW | 90,488 | 62,164 | N/A |
| LDTW | 90,488 | 62,164 | N/A |
| NDTW | 87,805 | 58,112 | N/A |
| Soft-DTW | **92,585** | 68,544 | N/A |
| OT | 60,488 | 42,261 | N/A |

FlashOPW có 1.891/2.050 query đúng ở k=1; OPW selected 1.839, TLp 1.893, Soft-DTW selected 1.898. MAP đo toàn ranking nên có thể có thứ tự khác ACC@1.

## Paired comparisons của selected

Các delta là **FlashOPW trừ baseline**, đơn vị điểm phần trăm.

| Baseline | Delta ACC@1 | Delta MAP | CI95% delta MAP | McNemar Holm p |
|---|---:|---:|---|---:|
| OPW journal | +2,537 | +3,369 | [3,008; 3,729] | 1,639e-6 |
| TLp | -0,098 | -0,486 | [-0,615; -0,357] | 1 |
| Soft-DTW | -0,341 | +1,402 | [0,650; 2,185] | 1 |
| DTW / LDTW | +1,756 | +7,782 | [6,975; 8,594] | 0,02569 |

- So với OPW selected: 78 query chỉ Flash đúng, 26 query chỉ OPW đúng; Flash có thêm 52 dự đoán đúng. Các thống kê ghi nhận chênh lệch rõ trong protocol này.
- So với TLp: Flash đúng riêng 13 query, TLp đúng riêng 15; chưa phát hiện khác biệt ACC@1 có ý nghĩa. Điều này không chứng minh tương đương. CI MAP âm cho thấy TLp retrieval tốt hơn trong ước lượng query bootstrap này.
- So với Soft-DTW: chưa phát hiện khác biệt ACC@1 có ý nghĩa; Flash có MAP cao hơn.
- CI MAP là paired percentile bootstrap 10.000 resamples và **chưa hiệu chỉnh nhiều phép so**. Holm áp dụng riêng cho McNemar ACC@1, 10 baseline/profile.
- CI/p-value có điều kiện theo gallery/selection cố định, giả định query độc lập; không đo biến động do retraining hoặc các nhóm query phụ thuộc.
- Các phép so có Flash được artifact gắn `convergence_valid=false`, vì giữ nguyên các cặp vượt tolerance dù rất sát ngưỡng. Không bỏ quality flag sau khi đọc TEST.

## Selected và preset: tuning không cải thiện đồng đều

| Metric | Preset ACC@1 % | Selected ACC@1 % | Preset MAP % | Selected MAP % |
|---|---:|---:|---:|---:|
| FlashOPW | 77,122 | 92,244 | 53,098 | 69,946 |
| OPW journal | 92,829 | 89,707 | 71,108 | 66,576 |
| OPW-KL | 92,829 | 89,659 | 71,110 | 66,503 |
| Soft-DTW | 91,366 | 92,585 | 64,757 | 68,544 |
| Sinkhorn | 61,073 | 60,732 | 42,651 | 42,307 |
| TCOT | 49,415 | 63,951 | 40,498 | 45,470 |

TLp, DTW/LDTW/NDTW và OT có selected/preset trùng nhau.

FlashOPW tuning tăng **15,122 điểm ACC** và **16,848 điểm MAP** so với preset. Ngược lại, OPW selected thấp hơn preset **3,122 điểm ACC** và **4,532 điểm MAP**. Suy luận phù hợp là selection trên TRAIN pilot nhỏ chưa bảo đảm generalization; dữ liệu hiện tại chưa xác định được nguyên nhân duy nhất.

OPW journal preset cao hơn FlashOPW selected **0,585 điểm ACC** và **1,162 điểm MAP**, nhưng có cặp chưa hội tụ như dưới đây. Hai profile phải được báo riêng; không đổi selection hoặc chọn bộ thắng sau khi xem TEST rồi gọi đó là TRAIN-selected.

## Vì sao completed_with_nonconvergence?

| Cấu hình | Số cặp vượt tau / 410.000 | Residual lớn nhất | Max iterations |
|---|---:|---:|---:|
| FlashOPW selected | 5 (0,00122%) | 0,00100002251565 | 1050 |
| FlashOPW preset | 2 (0,00049%) | 0,00100002810359 | 300 |
| OPW journal preset | 333 (0,08122%) | 0,00207300996408 | 4000 |
| OPW-KL preset | 322 (0,07854%) | 0,00206772796810 | 4000 |

Các selected entropic baseline còn lại có 0 cặp vượt tau. Phương pháp không entropic có N/A, không xem placeholder 0 là chứng nhận hội tụ.

Flash selected vượt tau nhiều nhất khoảng **2,25e-8**; mọi cặp dừng trước cap. Điều này phù hợp với sai khác FP32 giữa phép giảm stopping `P@1` và diagnostic `P@[1,Y]` đã sửa reader trước đây. Đây là suy luận từ diagnostics tổng hợp và cơ chế đã kiểm tra, chưa xem riêng ma trận của từng cặp trong gói upload.

OPW/OPW-KL preset có residual lớn nhất khoảng hai lần tolerance và có cặp ở cap; chất lượng hội tụ cần lưu ý hơn sai khác sát ngưỡng Flash. Các cặp vẫn được giữ trong MAP/ACC; không tự tăng cap, nới tolerance hay loại query sau TEST. Artifact không cung cấp phân bố từng cặp trong gói nhỏ nên chưa phân loại chính xác mọi cặp vượt tau theo stopping iteration.

## Thời gian đã ghi: không phải benchmark nhóm 5

| Cấu hình | Solve wall time tổng |
|---|---:|
| FlashOPW selected | 4,422 giờ |
| OPW selected | 0,454 giờ (~27,3 phút) |
| TLp | 1,291 giờ |
| OT CPU | 16,085 giờ |

Tổng 17 cấu hình duy nhất: **38,697 giờ**, tính từ solve wall time của từng job đã lưu, không phải elapsed terminal gồm mọi pause/resume/audit.

Ở workload độ dài 131 này, runner Flash từng cặp chậm hơn dense GPU batched OPW/TLp trong wall time đã ghi. Không thể dùng số này để tuyên bố FlashOPW có lợi thế tốc độ. Thời gian gồm conversion/diagnostics/compilation; khác batching, tham số và số vòng hội tụ. Nhóm 5/6 vẫn cần benchmark đúng protocol trên chuỗi dài, kiểm tra cả cùng công thức và khác metric, warmup, synchronization, bộ nhớ và end-to-end.

## Kết luận có thể dùng trong báo cáo

> Trên official TEST FacesUCR với tham số đóng băng từ TRAIN, FlashOPW đạt ACC@1 92,244% và MAP 69,946%. Nó cao hơn OPW journal selected lần lượt 2,537 và 3,369 điểm phần trăm; gần TLp về ACC nhưng thấp hơn TLp về MAP. OPW preset vẫn đạt điểm cao hơn FlashOPW selected, nên chưa kết luận ưu thế tổng quát. Run hoàn tất, server audit và local readback pass; giữ nguyên quality flag cho các cặp vượt tolerance. Chưa có bằng chứng lợi thế tốc độ từ nhóm 4.

Nhóm 4 đã hoàn thành workload và kiểm tra kết quả của protocol repo. Điều đó chưa chứng minh tái hiện đầy đủ mọi bảng trong journal, tương đương code tác giả hay hoàn thành benchmark tốc độ nhóm 5/6.

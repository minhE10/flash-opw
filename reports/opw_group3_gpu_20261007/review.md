# Nhóm 3 GPU: tuning công bằng trên TRAIN

Ngày audit: **07/10/2026**. Run server đã hoàn tất **138/138 job**, tất cả
11 metric có candidate hợp lệ để freeze. Đây là pilot selection trên TRAIN,
chưa phải đánh giá TEST hoặc tái hiện các bảng đầy đủ của journal.

RTX5080, GPU vật lý1 (`cuda:0` trong process), Torch2.11.0+cu128, hai CPU
thread, memory fraction0.45. FlashOPW dùng streaming IEEE FP32; các baseline
entropic dùng dense Torch FP32; DTW/Soft-DTW/LP dùng CPU. Code là implementation
trong repo, không phải code tác giả.

## Kiểm tra kết quả

Archive ở commit `c3bdc56` chứa289 entries, gồm directories, với138 NPZ,
138 job JSON và các manifest/report. SHA256:
`8644571c506ab4dc0b41e4e8fb3232a14f1bfc7769152ae3a7b5145897b9efa4`.
Nguồn run là commit `76a12f6`. Cả18 source hashes được ghi nhận khớp repo;
`git_dirty=true` không đi kèm thay đổi ở các source đã hash. Không suy đoán
nguyên nhân dirty ở những file ngoài danh sách này.

Đã chạy lại auditor local và tính lại **138 ma trận**: per-query AP,
predictions, MAP, ACC@1, SHA256 NPZ, labels/indices, residuals/iterations,
mean/SD ba split, eligibility và lựa chọn candidate. Cả ba CSV
candidate/preset/tuned khớp. TRAIN fingerprint khớp dataset đã dùng ở nhóm1–2.
Mỗi split gallery28/query28 không giao nhau, uniform marginals, ground cost
squared Euclidean, input archive làm tròn FP32, không fit preprocessing trên TEST.

Ba seed20261006/20261007/20261008 tạo **84 lượt validation**, gồm **71 query
TRAIN khác nhau**. Các split có thể dùng lại sample, không phải84 quan sát
độc lập. Bảy metric có tham số thử6 candidate; DTW/LDTW/NDTW/OT có1 candidate.
Tổng108192 cặp. Chọn mean ACC@1 trước, mean MAP phá hòa; ACC so ở12 chữ số
thập phân để tránh roundoff của cùng số correct count.

Đã giải lại **36 score mẫu** bằng oracle SciPy FP64 độc lập: các cặp có median
và max iteration của từng selected entropic metric trên mỗi split, với cùng
giá trị input FP32 và **cùng số vòng GPU đã dùng**. Tất cả đạt
rtol3e-3/atol3e-4; max absolute error6.609e-5 (Sinkhorn), riêng Flash1.023e-6.
Đây là sampled score verification, không phải full-matrix/coupling GPU parity.
Không chạy lại CUDA trên máy local. Xem [audit.json](audit.json) và
[sampled_scipy_oracle.json](sampled_scipy_oracle.json).

## Preset và tuned trên cùng protocol

Số liệu là **mean % của ba split TRAIN**. Hai bảng đầy đủ, kèm sample SD, ở
[comparison_report.md](comparison_report.md). Preset dùng tham số journal
của repo nhưng cùng stopping policy mới; không dùng số vòng20/100 của journal.

| Metric | Preset ACC@1 | Preset MAP | Tuned ACC@1 | Tuned MAP |
|---|---:|---:|---:|---:|
| FlashOPW | 58.333 | 58.464 | **79.762** | **74.405** |
| TLp | 80.952 | 75.130 | **80.952** | **75.130** |
| Soft-DTW | 72.619 | 69.106 | 78.571 | 72.956 |
| OPW journal | 77.381* | 72.899* | 76.190 | 71.397 |
| OPW-KL journal | 77.381* | 72.899* | 76.190 | 71.376 |
| DTW | 67.857 | 65.688 | 67.857 | 65.688 |
| LDTW | 67.857 | 65.688 | 67.857 | 65.688 |
| NDTW | 64.286 | 61.321 | 64.286 | 61.321 |
| TCOT | 35.714 | 44.922 | 46.429 | 49.580 |
| Sinkhorn | 39.286 | 47.527 | 41.667 | 47.268 |
| OT | 41.667 | 47.484 | 41.667 | 47.484 |

`*` Preset không đủ điều kiện freeze do một cặp chưa đạt ngưỡng ở cap.
Các score/cặp này vẫn được giữ khi tính bảng preset; không bỏ sample khó.
DTW/LDTW/NDTW/OT không có hyperparameter search trong protocol này, nên hai
cột không đổi. LDTW chia DTW cho query length; mọi query dài131 nên ranking
và chất lượng trùng DTW.

## Insight về FlashOPW và score

**Tuning prior cải thiện rõ kết quả pilot của FlashOPW**: ACC tăng21.429
điểm phần trăm, MAP tăng15.941 điểm phần trăm so với default, tại cùng ngưỡng
marginal. Candidate được chọn có `mu=50,epsilon=.1`,
`lambda1=1,lambda2=.1,sigma=.031943828249996996,cost_scale=1`, vẫn dùng
literal Eq.19. Đây cũng là tham số Flash đã tìm được ở tuning cũ, nay được
kiểm tra trên ba split thay vì một split. Không sửa default solver.

Candidate mu50/epsilon.3 có cùng mean ACC nhưng MAP73.179, thấp hơn74.405
của epsilon.1, nên quy tắc MAP phá hòa đã chọn candidate1. Mu200/epsilon.1
chỉ đạt ACC77.381/MAP71.562; trong grid đã thử, tăng prior thêm không cải thiện.
Grid sáu điểm là search nhỏ, không phải bằng chứng tìm được global optimum.

**FlashOPW rất gần TLp về chất lượng ở pilot này**, nhưng chưa có bằng chứng
ngang nhau trên TEST. TLp giữ weight50/epsilon.1, đạt hơn Flash1.190 điểm ACC
và0.725 điểm MAP. Tổng correct là67/84 cho Flash và68/84 cho TLp.

| Seed | Flash ACC@1 | TLp ACC@1 | Flash MAP | TLp MAP |
|---|---:|---:|---:|---:|
| 20261006 | 92.857 | 92.857 | 80.943 | 83.096 |
| 20261007 | 64.286 | 67.857 | 66.482 | 66.885 |
| 20261008 | 82.143 | 82.143 | 75.789 | 75.408 |

Hai metric có cùng cost giải `D+50F` và epsilon, nhưng Flash rank bằng Eq.19,
TLp bằng `<P,D+50F>`. Có4 NN-index disagreements trên84 lượt, nhưng chỉ1
lượt đổi đúng/sai; các trường hợp còn lại khác neighbor trong cùng class
hoặc cả hai vẫn sai. Full ranking chỉ trùng1/84 lượt. Có20/2352 cặp dừng
khác checkpoint giữa hai backend; do đó không gọi mọi coupling hữu hạn là
identical chỉ từ identity của cost.

Để tách yếu tố score, đã tái dựng query TRAIN index58, seed20261007,
gallery28 bằng SciPy FP64 tại số vòng Flash đã dùng cho từng cặp. **Cả hai
score tính từ cùng coupling/potentials, cùng backend/cost/iterations**:
Eq.19 chọn gallery row26, dự đoán class2; `<P,D+50F>` chọn row7, dự đoán
class6, đúng nhãn query. Cả hai NN khớp kết quả GPU tương ứng;
max Eq.19 score error so GPU1.240e-6. Vì giữ nguyên cost affine/Taylor trong
đối chứng này, chênh ACC đó không cần Taylor để giải thích.
Đây là diagnostic cho một query, không đổi score chính hoặc file selection.
Chi tiết: [score_control.json](score_control.json).

**Độ biến thiên split lớn**: Flash ACC sample SD14.434 điểm phần trăm,
TLp12.542; riêng Flash dao động64.286–92.857. Chênh mean1.190 điểm giữa hai
metric nhỏ so với variability này. Không dùng SD như confidence interval,
không kết luận thắng/thua thống kê từ validation sau search.

## Hội tụ ảnh hưởng trực tiếp selection

Stopping: max(row_L1,col_L1)<=1e-3, check mỗi50 vòng, cap4000, f rồi g.
Mọi selected entropic candidate đạt ngưỡng trên toàn2352 cặp/metric.
Có **ba candidate bị loại**, mỗi candidate một cặp ở seed20261008:

| Metric / candidate | Tham số liên quan | Residual tại4000 |
|---|---|---:|
| OPW / 0 | lambda1=1,lambda2=.1,sigma=1 | 0.001047856 |
| OPW-KL / 0 | lambda2=.1,sigma=1 | 0.001043756 |
| Sinkhorn / 5 | epsilon=.003 | 0.001477377 |

Vì preset OPW/OPW-KL bị loại, **tuned của chúng có thể thấp hơn preset** dù
search chứa preset. Candidate thắng phải đạt cả quality và convergence rule.
Sinkhorn epsilon.003 có cùng ACC với epsilon.01 và MAP cao hơn nhẹ, nhưng
bị loại vì cap. Không quy các hiện tượng này thành bug chọn candidate hoặc
khẳng định metric kém ở mọi nghiệm hội tụ.

Candidate hợp lệ: Flash/TLp/Soft-DTW/TCOT6/6, OPW/OPW-KL/Sinkhorn5/6.
Ngân sách bằng số attempt, không bằng số candidate hợp lệ hoặc thời gian.
Muốn kiểm tra độ nhạy cap, cần một experiment riêng khai báo cap mới;
không thay điều kiện để reselect theo TEST.

Flash selected có median350 vòng trên cả ba split, max600/650/650.
OPW selected có median125/150/150, max200; OPW-KL median100/150/150,
max200. Đây là stopping statistics theo cost khác nhau, không phải speedup.
Không thay adaptive policy bằng fixed200 cho evaluation tiếp theo.

Soft-DTW chọn gamma1 thay cho.1, tăng5.952 điểm ACC và3.849 điểm MAP.
TCOT chọn lambda3; Sinkhorn chọn epsilon.01, tăng ACC nhưng MAP thấp hơn
preset vì ACC là mục tiêu ưu tiên. Tuning không đảm bảo đồng thời tăng mọi metric.

## Freeze, giới hạn và bước tiếp theo

File dùng cho các experiment tiếp theo:
[selected_all_metrics.json](selected_all_metrics.json). Giữ score convention,
dataset fingerprint và stopping policy trong file; không truyền nó cho
`opw_knn --flash-parameters` vốn đọc schema fixed-iteration Flash-only cũ.
Sigma OPW journal và sigma Flash tiếp tục khác đơn vị; không copy chéo.

Sum recorded job times là1887.862 giây (~31.46 phút), bao gồm solve,
conversion trong operation, diagnostics/evaluation và compilation khi có.
Không gồm toàn bộ setup/file I/O; không gọi đây là end-to-end wall time hoặc
so tốc độ kernel. Flash pair-by-pair, dense baselines batching và CPU methods
có workload khác nhau. So runtime đúng protocol thuộc nhóm5–6.

Không so trực tiếp percentages này với TEST pilot64 trước đó: khác split,
gallery/query, số seed và iteration policy. Validation sau search có selection
bias; cần TEST độc lập để kết luận độ chính xác thực tế. Native length131,d1
ở FacesUCR chưa xác nhận lợi thế trên chuỗi dài hoặc nhiều feature.

**Nhóm3 GPU pilot hoàn tất và đã audit.** Theo plan, tiếp theo chạy nhóm5
pilot cùng GPU để ước lượng chi phí, rồi nhóm4 TEST đầy đủ với tham số đã
freeze; tiếp đến nhóm6 end-to-end. Không tự sửa score hoặc tune theo TEST.

Archive được bỏ khỏi `main` sau khi đọc, theo yêu cầu người dùng; giữ báo cáo
gọn và file freeze tại folder này. Bản nguyên vẹn còn ở Git history commit
`c3bdc56` và `outputs/opw_group3_gpu_import_20261007/original_results.tar.gz`
local (ignored), cùng thư mục đã giải nén. Xóa khỏi main không thu hồi dung
lượng lịch sử Git; không rewrite history.

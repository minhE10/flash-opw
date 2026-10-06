# Nhóm 2 GPU: prior, score và Taylor

Ngày đọc/audit: 06/10/2026. **Nhóm 2 đã hoàn tất pilot CPU và GPU.**
Kết quả do người dùng chạy trên RTX5080, GPU vật lý1 (`cuda:0` trong process),
Torch2.11.0+cu128, IEEE FP32, hai CPU thread, memory fraction.45.
Flash chạy hai model affine; bốn model inverse exact là dense Torch CUDA,
không phải kernel Flash cho inverse exact. Không đo tốc độ trong experiment này.

## Kiểm tra artifact

Archive người dùng push tại commit `a1785d9` chứa42 file: sáu JSON/CSV và36
NPZ. Run source commit `fdd4c29`; trạng thái `completed`,
`flash_verification=executed`, `numerical_parity_failures=0` và14 chunk hoàn tất.

Đã kiểm tra ZIP CRC; đối chiếu SHA256 source với repo, frozen selection,
TRAIN fingerprint, input hash và chỉ số query/gallery với CPU run local.
Tất cả khớp. Metadata ghi `git_dirty=true`, nhưng source hashes của các module
đã ghi nhận khớp; không suy đoán nguyên nhân dirty ngoài các hash đó.

Đã tính lại **108 hàng đánh giá** từ36 NPZ bằng evaluator của repo, bao gồm
per-query AP, predictions và ACC k1,3,5,7,15. Tất cả khớp evaluations JSON,
summary và MAP/ACC của CSV; mọi trường numeric của CSV cũng khớp summary.
Chỉ số query/gallery không giao nhau; mọi score hữu hạn.
Đã kiểm tra cả12 đối chứng fixed iterations (sáu model×100/200): ba score đều
đạt rtol3e-3,atol3e-4; coupling relative L2 được lưu đều dưới5e-3.
Max coupling error ở fixed iterations là2.257e-5, tại affine tuned200.

**Toàn bộ ranking CPU/GPU trùng nhau trong cả54 phép so**
(sáu model×ba mode×ba score), không chỉ nearest neighbor hay MAP/ACC.
Coupling error là diagnostic được runner lưu; audit không chạy lại CUDA hoặc
khôi phục P đầy đủ từ artifact. Kiểm tra MAP/ACC/ranking dùng score matrices gốc.
Chi tiết: [audit.json](audit.json), [ablation_results.csv](ablation_results.csv),
[run_state.json](run_state.json).

## Chất lượng tại cùng ngưỡng dừng

FacesUCR TRAIN,14 query×16 gallery; cùng pilot nhóm1 và CPU nhóm2. Giữ lambda1=1,
epsilon=.1,cost_scale=1; default sigma1, tuned sigma.031943828249996996.
Mỗi coupling đánh giá ba score; không solve lại chỉ để đổi score.

Tất cả224 cặp của cả sáu model trên cả hai backend đạt
max(row_L1,col_L1)≤1e-3 trong mode `residual_stop`, không có capped pairs.
Đây là ngưỡng marginal đã khai báo, không phải chứng nhận optimal objective gap.

Mỗi ô là **MAP% / ACC@1% trên GPU**, CPU cho cùng giá trị:

| Model | Dual theo quy ước Eq.19 | Spatial <P,D> | Affine <P,D+muF> |
|---|---:|---:|---:|
| affine default | 63.770 / 57.143 | 54.924 / 35.714 | 64.290 / 57.143 |
| affine tuned | 72.058 / 64.286 | 73.214 / 64.286 | 72.692 / 64.286 |
| inverse exact relative default | 64.335 / 57.143 | 54.860 / 35.714 | 64.335 / 57.143 |
| inverse exact relative tuned | 72.058 / 64.286 | 73.214 / 64.286 | 72.692 / 64.286 |
| inverse exact journal default | 60.156 / 42.857 | 77.289 / 71.429 | 77.289 / 71.429 |
| inverse exact journal tuned | 50.950 / 42.857 | 50.950 / 42.857 | 50.950 / 42.857 |

Score thứ ba dùng affine evaluation cost cho mọi model, không phải total cost
của journal. Main PDF vẫn là affine với Eq.19; exact dual dùng cùng quy ước f/g
để làm đối chứng. Không đổi default/selection từ kết quả pilot này.

## Kết luận từ đối chứng từng yếu tố

**Prior.** Giữ affine và Eq.19, mu1.05→50 tăng MAP8.288 điểm phần trăm,
ACC7.143 điểm phần trăm (8/14→9/14 đúng). Đây là kết quả TRAIN pilot; gallery
và query khác pilot64×64 TEST, không dùng mức64.286% ở đây để kết luận kết quả
85.938% trước đã giảm.

**Taylor.** Với tuned và cùng relative prior, inverse exact có cùng MAP/ACC
ở cả ba score. Dual và affine score có ranking trùng14/14 query; spatial có
ranking trùng13/14 nhưng cùng MAP/ACC. Max coupling difference exact-vs-affine
ở residual stop khoảng1.040%; coupling không đồng nhất.
Với default, inverse exact tăng dual MAP.565 điểm phần trăm, ACC giữ nguyên,
max coupling difference khoảng18.410%. Vì vậy không coi Taylor là vô hại
trên mọi tham số, nhưng **GPU pilot tuned không cho thấy Taylor làm giảm
MAP/ACC**. Kết luận này khớp CPU.

**Score.** Cùng coupling affine tuned sau dừng, spatial MAP cao hơn Eq.19
1.156 điểm phần trăm, affine/TLp cao hơn.635; ACC cả ba vẫn9/14. Với inverse
journal default, đổi dual sang spatial tăng MAP17.134 và ACC28.571 điểm phần
trăm trên cùng P. Cách tính score có ảnh hưởng rõ, nhưng không có một score
thắng trên mọi model. Việc chọn score cho ứng dụng cần validation đã khai báo.

**Prior journal.** Giữ inverse exact và spatial score, đổi relative Gaussian
sang prior vuông góc journal tăng MAP22.430 điểm phần trăm ở default,
nhưng giảm22.264 ở tuned. Sigma giữa hai định nghĩa có đơn vị khác nhau.
Với131×131, hệ số Gaussian theo F của profile tuned là49 ở relative và
420444.5 ở journal: sao chép sigma làm prior mạnh hơn rất nhiều, gần ép diagonal.
Đổi đúng đơn vị cho một cặp là
sigma_journal=sigma_relative/sqrt(1/N²+1/M²), tương ứng≈2.959 cho sigma tuned
hiện tại. Đây là quan hệ giải tích và đã có unit test, không phải tham số mới
được chọn qua TEST. Với chuỗi dài biến đổi, hệ số phụ thuộc từng cặp.

**Iterations và TLp.** Tuned có cùng D+50F,epsilon.1 với TLp preset.
Ở200 vòng, score TLp MAP71.502%, còn Eq.19 MAP72.058%.
Sau đạt ngưỡng, TLp MAP72.692%: tăng1.190 điểm phần trăm so với200 vòng,
trong khi ACC vẫn64.286%. Do đó NN/ACC ổn định không chứng minh MAP hay
coupling ổn định. Đổi score có thể đảo thứ tự MAP giữa fixed200 và residual stop.

## Hội tụ và precision

| Model GPU | Median checkpoint dừng | Checkpoint dừng lớn nhất | Max marginal L1 khi dừng |
|---|---:|---:|---:|
| affine default | 100 | 200 | 9.80e-4 |
| inverse relative default | 100 | 200 | 9.85e-4 |
| affine tuned | 300 | 600 | 9.97e-4 |
| inverse relative tuned | 300 | 600 | 9.99e-4 |
| inverse journal default | 1775 | 3800 | 1.00e-3, chưa làm tròn: .000999823 |
| inverse journal tuned | 50 | 50 | 9.12e-7 |

Kiểm tra mỗi50 vòng nên đây là checkpoint đầu quan sát, không phải số vòng
tối thiểu chính xác. Max600 affine tuned phù hợp nhóm1 quét thưa500/1000:
1000 là checkpoint đầu toàn bộ đạt trong sweep cũ. Không dùng iteration count
để báo speedup; journal tuned đang giải một prior gần diagonal khác hẳn.

Chỉ một cặp inverse journal default dừng khác CPU50 vòng; các model còn lại
có iteration matrix trùng CPU. Coupling relative L2 adaptive max1.126e-3 của
journal default bao gồm ảnh hưởng dừng ở iterations khác nhau; không diễn giải
thành lỗi fixed-iteration. Ranking CPU/GPU vẫn trùng toàn bộ.

## Lưu dữ liệu và bước tiếp theo

Bản ZIP gốc và42 file đã được giữ local trong
`outputs/opw_group2_gpu_import_20261006/` (Git ignore). File ZIP được bỏ khỏi
phiên bản hiện tại của repo sau audit theo phương án người dùng đã chọn.
Git history vẫn giữ [archive tại commit a1785d9](https://github.com/minhE10/flash-opw/blob/a1785d9/reports/opw_group2_gpu_20261006/results.zip).
SHA256 archive gốc ghi trong [audit.json](audit.json). Report gọn giữ CSV108
hàng, run_state và kiểm tra audit; không cần commit lại toàn bộ NPZ/JSON gốc.

**Nhóm2 hoàn tất pilot.** Chưa tái hiện bảng journal hoặc chứng minh chất lượng
TEST đầy đủ:14 query TRAIN đã liên quan tuning, một query đổi ACC7.143 điểm
phần trăm. Không suy rộng thành ưu thế thống kê hay Taylor không gây hại trên
mọi dữ liệu/parameters. Không có số timing để kết luận tốc độ.

Nhóm3 tiếp theo: tuning công bằng từng metric, phân biệt đơn vị prior và
freeze score/iteration policy trước TEST. Sau đó nhóm4 xác nhận MAP/ACC trên
dữ liệu TEST đầy đủ; nhóm5 so tốc độ cùng GPU và cùng tiêu chí hội tụ.

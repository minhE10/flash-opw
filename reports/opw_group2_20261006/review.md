# Nhóm 2: kết quả ablation CPU FlashOPW

Ngày chạy: 06/10/2026. **CPU FP64 đã hoàn tất; Flash CUDA cần chạy trên server.**
[Protocol và lệnh](../../docs/opw_group2.md),
[kế hoạch nhóm1–6](../../docs/opw_experiment_plan.md).

## Phạm vi và kiểm soát

FacesUCR TRAIN, 14 query×16 gallery, seed20261006, cùng chỉ số pilot nhóm1.
Chỉ số query/gallery không giao nhau. Đây là holdout khám phá đã liên quan tới
tuning trước đó; không phải TEST độc lập hay kết quả của pilot64×64 trước.
Một query đúng/sai làm ACC@1 đổi7.143 điểm phần trăm. Không suy rộng kết quả
nhỏ này thành ưu thế thống kê trên đầy đủ dataset hoặc journal.

Sáu model: affine, inverse exact relative, inverse exact với prior vuông góc
journal, mỗi model dùng hai profile default/tuned. Lambda1=1, epsilon=.1,
cost_scale=1 được giữ chung; sigma=1 hoặc.031943828249996996. Mu affine=1.05/50.
Giữ uniform marginals, alternating f rồi g, input FP32 được đưa cùng giá trị
vào oracle FP64. Cost q0 được loại khi solve và khôi phục khi tính dual.

Mỗi coupling có ba score: dual theo quy ước Eq.19, <P,D>, <P,D+mu*F>.
Với exact controls, score thứ ba vẫn dùng cùng **affine evaluation cost** để
kiểm soát phép so; không gọi nó là total cost của model journal. Coupling
không được solve lại chỉ để đổi score. Công thức chính vẫn main PDF affine
với Eq.19, không đổi solver hoặc artifact selection sau khi xem kết quả.

Đánh giá fixed100/200 và mỗi cặp dừng ở checkpoint50 đầu tiên đạt
max(row_L1,col_L1)≤1e-3, cap4000. **Tất cả224 cặp của cả sáu model đều đạt**;
không có capped pairs bị loại khỏi MAP/ACC. Không chứng nhận objective gap
từ ngưỡng marginal. Tổng54 hàng metric=6 model×3 chế độ×3 score; không phải
54 bộ coupling riêng. ACC có k1,3,5,7,15; k30 không hợp lệ với gallery16.

Lệnh CPU đã chạy:

```powershell
.venv-baselines/Scripts/python.exe -m experiments.opw_group2 --device cpu --output outputs/opw_group2_cpu_20261006
```

## MAP/ACC tại cùng tiêu chí dừng

Mỗi ô dưới đây là **MAP% / ACC@1%**, trên cùng query/gallery TRAIN:

| Model | Dual theo quy ước Eq.19 | Spatial <P,D> | Affine <P,D+muF> |
|---|---:|---:|---:|
| affine default | 63.770 / 57.143 | 54.924 / 35.714 | 64.290 / 57.143 |
| affine tuned | 72.058 / 64.286 | 73.214 / 64.286 | 72.692 / 64.286 |
| exact relative default | 64.335 / 57.143 | 54.860 / 35.714 | 64.335 / 57.143 |
| exact relative tuned | 72.058 / 64.286 | 73.214 / 64.286 | 72.692 / 64.286 |
| exact journal default | 60.156 / 42.857 | 77.289 / 71.429 | 77.289 / 71.429 |
| exact journal tuned | 50.950 / 42.857 | 50.950 / 42.857 | 50.950 / 42.857 |

## Insight từ các phép so từng yếu tố

**Prior strength.** Giữ cost affine và score Eq.19, đổi mu1.05→50 tăng
MAP8.288 điểm phần trăm và ACC7.143 điểm phần trăm (8/14→9/14 đúng).
Nearest neighbor thay đổi ở4/14 query, trong đó tổng số đúng tăng một.
Đây là hiệu ứng sigma/prior trên pilot, không phải ước lượng gain TEST.

**Taylor.** Với tuned, affine và inverse exact relative có cùng MAP/ACC ở cả
ba score sau khi đạt ngưỡng. Ranking của dual và affine score trùng hoàn toàn;
spatial full ranking trùng13/14 query nhưng MAP/ACC vẫn bằng nhau. Coupling
không đồng nhất: max relative L2 exact-vs-affine là1.040% tại residual stop,
1.044% tại200 vòng. Với default, max khác biệt coupling18.410%; dual MAP tăng
0.565 điểm phần trăm khi khôi phục inverse exact, ACC giữ8/14. Vì vậy không
coi Taylor là vô hại trên mọi tham số, nhưng **pilot tuned không cho thấy
Taylor làm giảm MAP/ACC**.

Theo công thức, C_affine-C_exact_relative=lambda1*F²/(1+F)≥0.
Mean <P,deltaC> trên coupling affine default là.007059, tuned là3.302e-5
(nhỏ hơn khoảng214 lần). Mean F giảm.04826→.002335; mean mass tại
|i/N-j/M|>.25 giảm23.37%→.0956%. Các số này hỗ trợ giả thiết coupling tuned
tập trung quanh đường chéo, nơi xấp xỉ tốt hơn. Weighted gap là diagnostic
cost trên coupling, không phải chứng nhận sai số optimal objective/MAP.

**Score.** Cùng coupling affine tuned sau dừng: đổi Eq.19 sang spatial tăng
MAP1.156 điểm phần trăm; sang affine/TLp score tăng.635 điểm phần trăm.
ACC vẫn9/14. Với exact journal default, cùng coupling nhưng score dual có
MAP60.156%, ACC42.857%, còn spatial có MAP77.289%, ACC71.429%.
Score có ảnh hưởng đáng kể và phụ thuộc model; không có một score thắng
trên mọi cấu hình. Chưa chọn score mới thay Eq.19 từ pilot này.

**Journal prior geometry.** Giữ inverse exact và spatial score: từ relative
sang prior vuông góc tăng MAP22.430 điểm phần trăm, ACC35.714 điểm phần trăm
ở profile default; nhưng giảm MAP22.264 và ACC21.429 ở profile tuned.
Sigma của hai định nghĩa có đơn vị khác nhau. Với N=M=131, hệ số Gaussian
trong cost theo F là429.025 thay vì.05 ở sigma1; ở sigma tuned là420444.5
thay vì49. Do đó không lấy trực tiếp sigma tuned relative để áp dụng journal
rồi kết luận model journal yếu: prior đã mạnh hơn rất nhiều, gần ép đường chéo.

Để giữ cùng Gaussian cost cho một cặp có chiều dài N,M, cần
sigma_journal=sigma_relative/sqrt(1/N²+1/M²). Với131×131, sigma relative tuned
tương ứng sigma journal≈2.959. Unit test kiểm tra cost centered khớp khi đổi
đúng đơn vị. Đây là quan hệ giải tích, không phải một candidate mới được
tuning/chọn. Với chiều dài biến đổi, hệ số chuyển đổi phụ thuộc từng cặp.

## Đối chứng TLp: score và iterations

Tuned affine có cùng cost D+50F,epsilon.1 với TLp preset. Unit test đối chiếu
<P,D+50F> ở100/200 vòng với TLp NumPy/SciPy độc lập, rồi đối chiếu coupling
và Eq.19 với OPW dense. Không dùng code tác giả.

| Vòng/chế độ | Eq.19 MAP | TLp/affine MAP | ACC@1 cả hai | Cặp đạt tau |
|---|---:|---:|---:|---:|
| fixed100 | 71.993% | 71.502% | 64.286% | 1/224 |
| fixed200 | 72.058% | 71.502% | 64.286% | 54/224 |
| residual stop | 72.058% | 72.692% | 64.286% | 224/224 |

Ở200 vòng, đổi score sang TLp làm MAP giảm.556 điểm phần trăm; sau dừng,
score TLp cao hơn Eq.19 .635 điểm phần trăm. TLp MAP100→200 không đổi nhưng
full ranking chỉ trùng8/14 query; từ200→residual stop MAP tăng1.190 điểm phần
trăm và NN giữ nguyên. Không dùng NN stability để kết luận MAP ổn định hoặc
coupling đã hội tụ. Mọi so sánh fixed iterations còn phải ghi residual.

## Số vòng đến ngưỡng và artifact

| Model | Median checkpoint dừng | Checkpoint dừng lớn nhất |
|---|---:|---:|
| affine default | 100 | 200 |
| exact relative default | 100 | 200 |
| affine tuned | 300 | 600 |
| exact relative tuned | 300 | 600 |
| exact journal default | 1775 | 3800 |
| exact journal tuned | 50 | 50 |

Đây là checkpoint quan sát mỗi50 vòng, không phải số vòng dừng chính xác hoặc
timing benchmark. Max600 của tuned phù hợp với nhóm1: sweep thưa trước đó
chỉ kiểm tra500 rồi1000 nên ghi1000 là checkpoint đầu toàn bộ đạt.
Journal tuned đạt nhanh trên cost gần diagonal khác hẳn; không xem đây là
speedup công bằng trên cùng bài toán.

[ablation_results.csv](ablation_results.csv), [contrasts.json](contrasts.json),
[evaluations.json](evaluations.json), [summary.json](summary.json),
[environment.json](environment.json), NPZ matrices, chunk caches và run_state
được lưu cùng report. Chunks giữ toàn bộ224 cặp, không lưu P đầy đủ.
Environment giữ fingerprints TRAIN, selection, input FP32 và source SHA.
Runner `experiments/opw_group2.py` SHA256 chuẩn hóa newline:
`f27ee20370307ca988714557902f4f63d63bdfaf89c7e4974545e45a951f20d6`.
Không đổi source số học trong khi run để giữ provenance khớp.

Validation: **77 passed,21 GPU skipped**, bash syntax đã đạt. Test kiểm tra
inverse cost/gap, geometry/đơn vị sigma, các score cùng P với SciPy oracle,
TLp identity, checkpoint sau adaptive stop, cap không được coi là hội tụ,
TRAIN-only/freeze/resume và dense FP32 so với FP64. GPU end-to-end test đã có.
`numerical_parity_failures=0` của CPU run không xác nhận Flash GPU, vì CUDA
chưa thực hiện trong run này.

## Bước tiếp theo trên server

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
bash scripts/validate_flash_opw.sh 1
bash scripts/run_opw_group2.sh 1 \
  --flash-parameters reports/opw_tuning_20261006/selected_parameters.json \
  --output outputs/opw_group2_gpu_v1
```

GPU run bổ sung Flash IEEE FP32 cho affine và dense CUDA FP32 cho exact;
so với FP64 tại fixed100/200. Chờ output server để xác nhận các ablation
trên GPU. Gửi `ablation_results.csv`, `contrasts.json`, `summary.json` và
`environment.json`, tốt nhất cả folder. Không cần chạy lại tuning hoặc TEST.
Nhóm3 tiếp theo cần tuning công bằng từng metric trong đúng đơn vị prior;
nhóm4 mới xác nhận chất lượng trên TEST đầy đủ và uncertainty.

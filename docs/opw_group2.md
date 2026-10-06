# Nhóm 2: ablation prior, score và Taylor

Mục tiêu: xác định phần nào làm thay đổi ranking/MAP/ACC khi so FlashOPW
với các metric khác. Công thức chính vẫn là main PDF affine, score Eq.19;
nhóm 2 không thay solver hoặc artifact hyperparameter đã freeze.

## Các đối chứng

Đặt F=(i/N-j/M)², kappa=epsilon/(2sigma²), mu=lambda1+kappa,
q0=epsilon*log(sigma*sqrt(2pi))-lambda1. Runner loại q0 khỏi cost khi solve,
khôi phục q0 khi tính dual score. Mỗi profile có ba cost:

| Tên | Cost không chứa q0 | Yếu tố được thay |
|---|---|---|
| affine | D+mu*F | Main PDF, đối chứng chính |
| exact_relative | D+kappa*F+lambda1*F/(1+F) | Khôi phục inverse exact, giữ relative Gaussian prior |
| exact_journal | D+kappa*F/(1/N²+1/M²)+lambda1*F/(1+F) | Chỉ đổi Gaussian prior sang khoảng cách vuông góc của journal |

Hai profile default và tuned đọc từ selection đã freeze. Runner yêu cầu
lambda1, epsilon và cost_scale giống nhau: chỉ sigma thay đổi. Với selection
FacesUCR hiện tại, mu=1.05/50 và lambda1=1, epsilon=.1.

Đối chứng Taylor so affine với exact_relative cùng profile. Đối chứng hình học
prior so exact_relative với exact_journal cùng profile. Không gộp hai thay đổi
này rồi quy toàn bộ chênh lệch cho Taylor.

Từ **cùng một coupling**, tính ba score:

* `dual_score`: literal Eq.19 cho affine; cùng biểu thức f/g cho exact controls,
  được gọi là dual theo quy ước Eq.19, không gọi là công thức affine của main.
* `spatial_score`: <P,D>, đúng quy ước ranking OPW journal của implementation.
* `affine_score`: <P,D+mu*F> cho mọi coupling, kể cả exact controls. Không thay
  score này thành cost solved của exact/journal vì sẽ đổi cả cost lẫn score.

`affine_tuned/affine_score` là TLp khớp tham số từ chính coupling affine tuned.
Với mu50,epsilon.1 nó trùng cost của preset TLp hiện tại. So các hàng sau:

1. tuned 200 vòng: dual_score vs affine_score — chỉ đổi score.
2. affine_score 100 vs200 vòng — chỉ đổi iterations, giống preset TLp100.
3. affine_score 200 vs residual_stop — ảnh hưởng chưa hội tụ.

Không solve lại coupling chỉ để đổi tên thành TLp. Unit test so kết quả alias
này với TLp NumPy/SciPy độc lập của repo. Nếu selection đổi mu/epsilon,
`matches_legacy_weight50_epsilon01` cho biết preset TLp còn trùng hay không.

## Dữ liệu, hội tụ và backend

Mặc định FacesUCR TRAIN: 14 query×16 gallery, seed20261006, cùng chỉ số pilot
nhóm1; không mở TEST. Split này đã được dùng trong quá trình chẩn đoán/tuning,
nên MAP/ACC chỉ là kết quả validation khám phá, không phải ước lượng TEST độc lập.
Ba score của mỗi model có cùng query/gallery, tie rules, toàn gallery MAP và
ACC k=1,3,5,7,15. Không chọn score hay hyperparameter thắng cuộc từ nhóm này.

Hai chế độ: fixed100/200; và mỗi cặp dừng ở checkpoint đầu tiên có
max(row_L1,col_L1)≤1e-3, kiểm tra mỗi50 vòng, cap4000. Có thể đặt tau1e-4 bằng
output mới. Những cặp chưa đạt ở cap vẫn có score hữu hạn và được đánh dấu;
không loại khỏi MAP/ACC. So sánh residual_stop có capped pairs là kết quả
tạm thời về lời giải, không phải đối chứng đã hội tụ cùng độ chính xác.
Ngưỡng marginal không bảo đảm sai số objective phổ quát.

CPU: tất cả sáu cost solve trực tiếp bằng Torch FP64 với cost tường minh.
CUDA: thêm Flash IEEE FP32 cho hai model affine; bốn model inverse exact dùng
dense Torch FP32 trên cùng GPU. Kiểm tra numerical parity với CPU FP64 tại
fixed iterations; adaptive stop có thể khác iteration giữa backend nên không
đòi coupling khớp tại hai thời điểm dừng khác nhau.

Diagnostic tính coupling relative L2 vs affine cùng profile và
`weighted_taylor_gap`=<P,lambda1*F²/(1+F)>; đây là sai số cost affine so với
inverse relative exact, luôn không âm. Tính thêm moment F và mass ngoài
|i/N-j/M|>.25 để kiểm tra coupling có tập trung quanh đường chéo không.
Weighted gap là chẩn đoán sai số cost trên coupling đang xét, không phải
certified gap của nghiệm tối ưu hay sai số MAP/ACC.
Coupling được materialize cho pilot: đây không phải benchmark bộ nhớ hoặc tốc
độ streaming. Các tham số được giữ cố định khi đổi loại cost.

## Chạy trên server

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
bash scripts/validate_flash_opw.sh 1
bash scripts/run_opw_group2.sh 1 \
  --flash-parameters reports/opw_tuning_20261006/selected_parameters.json \
  --output outputs/opw_group2_gpu_v1
```

GPU1 phải là allocation được cấp. Wrapper giữ một GPU và hai CPU thread;
memory fraction mặc định .45, dense working estimate128MiB. Mỗi chunk chứa
16 cặp và chạy cả sáu cost; không mở rộng gallery trước khi xem capped pairs
và thời gian pilot. Với gallery nhỏ hơn15, truyền `--ks` phù hợp.

CPU local:

```powershell
.venv-baselines/Scripts/python.exe -m experiments.opw_group2 --device cpu --output outputs/opw_group2_cpu_20261006
```

Output: `ablation_results.csv`, `contrasts.json`, `evaluations.json`,
`summary.json`, `environment.json`, `run_state.json`, NPZ của mọi model/mode
và chunks phục vụ resume. Contrasts có delta MAP/ACC tính bằng điểm phần trăm,
NN/full-ranking agreement và `both_satisfy_tau`. Row mode/backend phải khớp
khi diễn giải một ablation. Gửi toàn folder để giữ labels, chỉ số và fingerprint.

Resume: cùng lệnh thêm `--resume`, giữ nguyên code/data/settings/selection và
môi trường. Runner bỏ qua complete chunks; chunk dang dở được solve lại.
Không dùng chung output CPU/GPU. Numerical parity fail làm runner exit1;
capped controls được báo rõ và không bị ngụy trang thành kernel failure.

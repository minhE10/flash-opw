# Nhóm 1: correctness và hội tụ FlashOPW

Ngày chạy: 06/10/2026. **Phần CPU đã hoàn tất; Flash CUDA chưa được chạy tại máy local.**
Kế hoạch đầy đủ: [experiment 1–6](../../docs/opw_experiment_plan.md).

## Protocol đã thực hiện

* FacesUCR TRAIN, 14 query và 16 gallery có chỉ số không giao nhau: 224 cặp mỗi
  cấu hình. Seed `20261006`; không đọc TEST hoặc dùng accuracy để chọn tham số.
* Giữ hai cấu hình default và tuned từ
  [selection đã freeze](../opw_tuning_20261006/selected_parameters.json).
  Default: mu=1.05; tuned: mu=50. Cả hai epsilon=0.1, lambda1=1, cost_scale=1.
* Giữ công thức main PDF, relative time i/N,j/M, score Eq.19, uniform marginals,
  alternating f rồi g. Input được lượng tử FP32 một lần trước khi đưa cùng
  giá trị vào các backend. Oracle độc lập dùng cost D+mu*F trực tiếp, CPU FP64.
* Parity ở 20 và 200 vòng: 6 synthetic shape, thêm 4 cặp TRAIN mỗi cấu hình.
  Synthetic gồm 7×11,d3; 37×79,d65; 131×131,d1; 257×513,d13;
  17×29,d390; 1025×1537,d1. So score, spatial cost, coupling P và P@[1,Y].
* Convergence: 20/100/200/500/1000/2000 vòng trên cả 224 cặp mỗi cấu hình.
  Sai số biên là max(||P1-a||₁, ||Pᵀ1-b||₁); kiểm tra tau=1e-3 và 1e-4.
  NPZ giữ cả score matrix, spatial cost, mass, hai marginal và chỉ số TRAIN.
* CPU local: Torch 2.11.0+cpu, hai thread. Đây là kiểm tra số học/hội tụ,
  không phải benchmark tốc độ GPU hoặc đánh giá MAP/ACC trên TEST.

Lệnh đã chạy tại local:

```powershell
.venv-baselines/Scripts/python.exe -m experiments.opw_group1 --device cpu --output outputs/opw_group1_cpu_20261006
```

## Parity CPU: 52/52 đạt

Mỗi cấu hình có 20 kiểm tra dense FP32 và 6 kiểm tra tiled FP64 trên shape nhỏ.
Các giá trị trong bảng là lỗi lớn nhất trên các case của từng backend:

| Cấu hình | Backend | Lỗi tuyệt đối Eq.19 | P relative L2 | P@[1,Y] relative L2 |
|---|---|---:|---:|---:|
| default | dense FP32 | 1.03e-7 | 5.51e-6 | 2.84e-6 |
| tuned | dense FP32 | 6.71e-7 | 3.90e-5 | 3.80e-5 |
| default | tiled FP64 | 1.11e-16 | 1.99e-15 | 8.64e-16 |
| tuned | tiled FP64 | 6.22e-15 | 1.03e-13 | 2.68e-14 |

Ngưỡng khai báo: FP32 score/spatial rtol=3e-3, atol=3e-4; coupling/apply
relative L2≤5e-3. FP64 score/spatial rtol=1e-10, atol=1e-11; coupling/apply
relative L2≤1e-10. Sai số quan sát nhỏ hơn đáng kể các ngưỡng này.
Parity không yêu cầu marginal hội tụ: hai backend được so ở cùng số vòng.

## Hội tụ CPU FP64

| Cấu hình | Vòng | Sai số biên lớn nhất | Cặp đạt ≤1e-3 | Cặp đạt ≤1e-4 | Max score drift so với 2000 vòng |
|---|---:|---:|---:|---:|---:|
| default | 200 | 3.90e-4 | 224/224 | 209/224 | 9.47e-6 |
| default | 500 | 8.54e-5 | 224/224 | 224/224 | 2.11e-6 |
| default | 2000 | 1.93e-5 | 224/224 | 224/224 | 0, theo định nghĩa |
| tuned | 200 | 1.29e-2 | 54/224 | 1/224 | 1.81e-3 |
| tuned | 500 | 1.89e-3 | 218/224 | 72/224 | 7.61e-5 |
| tuned | 1000 | 2.74e-4 | 224/224 | 221/224 | 8.60e-6 |
| tuned | 2000 | 7.22e-5 | 224/224 | 224/224 | 0, theo định nghĩa |

* **200 vòng đủ cho default tại tau=1e-3 trên pilot này, nhưng chưa đủ cho tuned.**
  Tuned đạt toàn bộ tại checkpoint 1000 với tau=1e-3, và 2000 với tau=1e-4.
  Đây là checkpoint đầu quan sát đạt toàn bộ; không phải số vòng dừng chính xác.
* Tuned có prior mạnh hơn (mu50 thay vì mu1.05); trong cùng protocol, residual
  còn cao hơn ở 200 vòng. Vì vậy các so sánh tốc độ tiếp theo cần báo cáo cả
  fixed iterations và thời gian đến cùng sai số biên.
* Ở 200 vòng, cả default và tuned có nearest neighbor **và toàn bộ ranking**
  trùng checkpoint 2000 trên 14 query này. Điều đó cho thấy ranking pilot ổn
  định sớm hơn marginal. Không suy rộng thành MAP/ACC của TEST đã ổn định:
  với tuned, 2 query vẫn có NN margin nhỏ hơn hai lần sai số score quan sát.
* Mass xấp xỉ 1 không chứng minh từng marginal đúng. Primal-minus-dual được
  lưu để chẩn đoán; khi coupling chưa khả thi, đây không phải certified gap.
* Không có cặp nào chưa đạt hai ngưỡng tại cap 2000 trong tập TRAIN pilot này.
  Score drift=0 tại checkpoint cuối chỉ là so nó với chính nó.

## Artifact và phiên bản

[summary.json](summary.json), [parity.json](parity.json),
[convergence.csv](convergence.csv), [convergence.json](convergence.json),
[environment.json](environment.json), `run_state.json`, `reference_state.json`
và 12 NPZ reference được lưu cùng thư mục này. Environment giữ fingerprints
dữ liệu, selection, chỉ số TRAIN, packages và SHA256 source lúc bắt đầu chạy.

[runner_snapshot.py](runner_snapshot.py) giữ chính xác runner đã thực hiện CPU
run: SHA256 chuẩn hóa newline
`a05ee265cb0692313f5640611557c45df346d092021be484c5caf7da704b852b`, khớp
`environment.signature.sources`. Runner hiện hành thêm bước bỏ qua các parity
case đã hoàn tất trước khi tạo oracle khi resume; các phép tính và protocol
không đổi. Không chỉnh lại metadata gốc để giả khớp phiên bản mới.

Kiểm tra runner hiện hành và các phần OPW liên quan: **65 passed, 20 GPU tests
skipped** trên CPU; bash syntax check cho hai script mới/cập nhật đã đạt.
Unit/integration tests kiểm tra oracle incremental so với restart, hai marginal,
ranking khi margin nhỏ, TRAIN-only end-to-end, freeze artifact và resume từ chối
code/settings khác. GPU end-to-end test đã có nhưng cần server thực hiện.

## Phần còn lại: chạy Flash trên GPU server

```bash
cd /home/doanpt/minh.nd/flash-opw
git pull --ff-only origin main
conda activate minh
bash scripts/validate_flash_opw.sh 1
bash scripts/run_opw_group1.sh 1 \
  --flash-parameters reports/opw_tuning_20261006/selected_parameters.json \
  --output outputs/opw_group1_gpu_v1
```

GPU runner kiểm tra Flash IEEE FP32 với oracle CPU FP64: coupling/apply trên
các parity case và toàn score matrix của 224 cặp ở từng checkpoint. Nó đo
marginal Flash thực tế, không gán residual oracle cho Flash. Đây không phải
phép đo runtime; reference FP64 và diagnostics nằm trong quá trình chạy.
GPU wrapper dùng đúng một GPU được cấp, memory fraction mặc định 0.45 và hai
CPU thread. Nếu allocation hiện hành khác GPU1, giữ allocation đã được cấp.

Nếu gián đoạn, chạy lại cùng lệnh với `--resume`; không đổi code, dữ liệu,
selection hoặc môi trường. Dùng output mới khi thay settings/phiên bản. Gửi
folder `outputs/opw_group1_gpu_v1` để kiểm tra parity, convergence và rankings.
**Chỉ sau kết quả này mới kết luận phần correctness của Flash CUDA đã đạt.**

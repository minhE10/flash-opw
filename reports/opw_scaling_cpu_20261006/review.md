# Kiểm tra benchmark chuỗi dài — 06/10/2026

Đã chạy CPU local để kiểm tra runner và scaling; không có GPU CUDA trên máy
này nên chưa có timing FlashOPW/RTX5080. Không dùng các số CPU dưới đây để
kết luận speedup GPU, hay tái hiện runtime journal.

## Các lượt đã chạy

* Smoke nhỏ `16×20,d1,seed42`: cả 10 reference CPU journal hoàn tất,
  một warmup và một repeat; 7 metric GPU được ghi skipped đúng backend.
  Có kiểm tra tạo figure và resume không chạy lại các phép đo đã hoàn tất.
* Chuỗi dài: `n={512,1024,2048}`, `m=1.25n`, `d={1,13}`, seed42/43:
  12 cặp input. Mỗi cặp chạy DTW, soft-DTW, Sinkhorn và OPW: 48 case.
  Mỗi case một warmup và hai repeat, CPU FP64, threads2;
  Sinkhorn100 vòng, OPW20 vòng theo preset journal của repo.
* 37 case hoàn tất; 11 case timeout, không có code failure.
  Deadline45s cho toàn child process, gồm khởi động, warmup và hai repeat.
  Giá trị median/min/max chỉ có cho những case hoàn tất; timeout không có
  timing giả và không được dùng tính tỉ số.

Dữ liệu là chuỗi sin liên tục, warp1.15, noise.03, feature scale1/sqrt(d),
làm tròn FP32 trước khi đưa vào cả hai backend. Không phải sequence thật
của journal, không có MAP/ACC cho benchmark này. Xem raw
[timings.csv](long/timings.csv), [checks.json](long/checks.json),
[environment.json](long/environment.json), [run_state.json](long/run_state.json).
Một số phép đo chạy cùng thời gian với kiểm tra CPU khác trên desktop;
đây là kiểm tra chức năng/scaling, chưa phải performance run trên máy được
kiểm soát tải. Chưa có kiểm tra GPU nên chưa kết luận ưu thế tốc độ.

Runner lưu input SHA256, từng repeat và trạng thái checkpoint. Code GPU
baseline được bổ sung trong quá trình lượt CPU dài đang chạy; source hashes
của lượt dài lưu snapshot tại thời điểm bắt đầu. Các metric CPU trong lượt
này không đổi. Smoke sau cùng dùng snapshot có cả năm baseline entropic GPU.

## Kiểm tra code

59 test CPU pass, 19 test GPU skip trên máy local. Bao gồm chia validation
không giao gallery, train loader không cần TEST, fingerprint, lựa chọn ACC/MAP,
deduplicate hyperparameter, FP64 oracle với ragged lengths, long rectangular
257×513 và 1025×1537, cùng năm Torch baseline đối chiếu NumPy/SciPy.
Smoke k-NN có đủ 10 baseline, Flash CPU oracle mặc định và tuned, dense controls;
không có failure. Parity trong smoke cuối: max score error mặc định
7.11e-15, tuned8.88e-16, nearest-neighbor agreement100% cho cả hai.
Tuning, k-NN và scaling đều đã kiểm tra resume trên output phù hợp snapshot.
Bash syntax và Python compilation qua.

`scripts/validate_flash_opw.sh` sẽ yêu cầu và thực hiện 19 kiểm tra GPU trên server,
gồm các chuỗi dài với prior mạnh và năm baseline GPU. Bộ benchmark hỗ trợ
`--equal-entropic-iters 200` để đo cùng số vòng, diagnostics ngoài timing,
và dense memory preflight. Lượt thực hiện trên server trong
[hướng dẫn](../../docs/opw_tuning_scaling.md) là bước cần thiết để xác nhận tốc độ.

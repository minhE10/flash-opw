# Giải thích thuật ngữ trong báo cáo thực nghiệm

Tài liệu đi kèm `experiment_report_20261007.tex`, cập nhật ngày 08/10/2026. Các định nghĩa dưới đây ưu tiên nghĩa được dùng trong báo cáo và code của repository. Không liệt kê các khái niệm chuyên môn OT như coupling, marginals, dual potentials, Sinkhorn, OPW và các prior của chúng. Một số thuật ngữ toán học tổng quát như gradient, residual và Taylor vẫn được giải thích vì chúng xuất hiện ở nhiều lĩnh vực.

## 1. Dữ liệu và vai trò của các tập mẫu

| Thuật ngữ | Giải thích và ví dụ trong báo cáo |
|---|---|
| Dataset | Bộ dữ liệu dùng trong thí nghiệm. FacesUCR là dataset chuỗi thời gian chính của phần đánh giá thực. |
| Sample / mẫu | Một đối tượng dữ liệu. Trong FacesUCR, một sample là một chuỗi; trong benchmark point cloud, một điểm là một phần tử của đám mây điểm. Cần đọc theo ngữ cảnh. |
| Sequence / time series | Chuỗi các giá trị có thứ tự, thường theo thời gian. Length 131 nghĩa là chuỗi có 131 vị trí, không có nghĩa là 131 chuỗi. |
| Length / native length | Độ dài chuỗi / độ dài gốc trước khi đổi kích thước. “Native length 131” nghĩa là giữ 131 vị trí của dữ liệu gốc. |
| Feature / dimension / d | Thuộc tính tại một vị trí hoặc một điểm; số feature là dimension. Một vị trí có một giá trị thì d=1; có 13 giá trị thì d=13. |
| Spatial feature | Phần đặc trưng của dữ liệu trước khi thêm tọa độ thời gian. Tên này không bắt buộc dữ liệu phải là tọa độ địa lý hoặc ảnh. |
| Label / class | Label là nhãn gắn với sample; class là lớp mà nhãn biểu thị. FacesUCR có 14 class. |
| Ground truth | Nhãn hoặc đáp án chuẩn dùng để chấm kết quả. Trong phân loại, đó là label thật của query. |
| TRAIN | Phần dữ liệu dành cho phát triển/chọn tham số. Ở đây không nhất thiết có một mạng neural được huấn luyện: TRAIN còn được dùng làm gallery và chia holdout để tuning. |
| Validation | Tập hoặc bước kiểm tra để chọn cấu hình. “Validation split” là tập dữ liệu; “GPU validation” lại là kiểm tra implementation. Hai nghĩa cần phân biệt. |
| TEST / official TEST | Tập đánh giá sau khi đã chọn và đóng băng tham số / tập TEST chính thức của dataset. Không dùng nó để chọn tham số trong protocol chính. |
| Gallery | Tập mẫu để query tìm hàng xóm hoặc mẫu tương tự. Ví dụ gallery 200 là 200 chuỗi TRAIN được dùng làm danh sách đối chiếu. Không phải thư viện ảnh giao diện. |
| Query | Mẫu đang được đem đi so với toàn bộ gallery. Ví dụ 2050 TEST query nghĩa là lần lượt đánh giá 2050 chuỗi TEST. |
| Gallery/query 28/28 | Có 28 mẫu để đối chiếu và 28 mẫu cần dự đoán. Tính mọi cặp tạo 28×28=784 score, không phải 56 score. |
| Pair / cặp | Một query và một gallery sample, hoặc một cặp đầu vào cần so sánh. 2050×200=410000 cặp cho một cấu hình full TEST. |
| Split | Cách chia/chọn dữ liệu thành các tập có vai trò khác nhau. Đổi split có thể làm accuracy thay đổi dù thuật toán giữ nguyên. |
| Holdout | Giữ một phần dữ liệu ra ngoài tập dùng làm đối chiếu để đánh giá/chọn tham số. TRAIN holdout vẫn lấy từ TRAIN, không phải TEST chính thức. |
| Stratified | Chia mẫu có xét class để duy trì sự hiện diện/tỷ lệ lớp theo quy tắc của runner. Không đồng nghĩa mọi lớp đều có số mẫu bằng nhau. |
| Balanced subset / subset cân bằng | Tập con được chọn sao cho số mẫu các lớp bằng nhau hoặc gần nhau. |
| Disjoint / không giao nhau | Không dùng cùng một sample ở cả gallery và query của một split. |
| Seed / random seed | Giá trị khởi tạo bộ sinh số giả ngẫu nhiên. Giữ cùng seed và cùng quy trình giúp tái tạo việc chọn mẫu/input; seed riêng lẻ không bảo đảm mọi kết quả GPU giống từng bit. |
| Generator | Bộ sinh số giả ngẫu nhiên. Runner dùng nó để tạo input từ seed. |
| Index / indices | Số thứ tự dùng để xác định sample. “TRAIN index 58” là một mẫu cụ thể, không phải class 58. Index có thể bắt đầu từ 0 hoặc 1 tùy code. |
| Preprocessing | Xử lý dữ liệu trước thí nghiệm, như chuẩn hóa hoặc đổi kích thước. “Fit preprocessing” là ước lượng các thông số xử lý từ dữ liệu. |
| Resample dữ liệu | Lấy mẫu lại chuỗi để đổi số vị trí, chẳng hạn 131 thành 256. Khác với bootstrap resample ở mục thống kê. |
| Synthetic | Dữ liệu tự sinh bằng chương trình, không phải dữ liệu thu thập thực. |
| Point cloud | Tập các điểm trong không gian d chiều. Benchmark này có hai tập X và Y cần được so sánh. |
| Uniform [0,1]^d | Mỗi điểm nằm trong khối d chiều có từng tọa độ từ 0 đến 1 và được lấy theo phân phối đều. |
| Gaussian / Gaussian noise | Phân phối chuẩn / nhiễu được lấy từ phân phối chuẩn. Có thể hình dung nhiễu nhỏ cộng vào đường tín hiệu để dữ liệu bớt hoàn hảo. |
| Sin trajectory / smooth trajectory | Chuỗi dạng đường sin / đường thay đổi trơn. Dùng tạo dữ liệu synthetic cho kiểm tra scaling. |
| Time warp | Co giãn hoặc biến đổi trục thời gian của chuỗi. Hai tín hiệu có thể cùng hình dạng nhưng diễn ra nhanh/chậm khác nhau. |
| Data leakage / dùng TEST để chọn tham số | Thông tin tập đánh giá ảnh hưởng quá trình chọn phương pháp/tham số, làm đánh giá ít độc lập hơn. “Đã xem pilot TEST” phải được ghi rõ. |
| Blind TEST evaluation | Đánh giá trên TEST mà kết quả TEST chưa được dùng để định hướng phát triển/chọn cấu hình. Tên “official TEST” không tự chứng minh cả quá trình là blind. |

## 2. Xếp hạng và đánh giá chất lượng

| Thuật ngữ | Giải thích và ví dụ |
|---|---|
| Score / distance | Con số dùng so độ giống nhau của hai mẫu. Trong báo cáo, sắp tăng dần: số nhỏ được xếp gần hơn. Các phương pháp có định nghĩa score khác nhau nên không so giá trị thô như cùng một đại lượng. |
| Metric | Trong các bảng thí nghiệm, thường chỉ một phương pháp tạo distance/score như DTW, hoặc một chỉ số đánh giá như ACC/MAP. Tên “metric” ở đây không tự khẳng định distance thỏa mọi tiên đề metric toán học. |
| Distance matrix / score matrix | Bảng score của mọi query với mọi gallery sample. Full TEST có 2050 hàng × 200 cột. |
| Ranking | Danh sách gallery được sắp từ gần đến xa cho một query. |
| Retrieval | Tìm và xếp hạng những mẫu liên quan đến query trong gallery. |
| NN — nearest neighbor | Hàng xóm gần nhất: gallery sample đứng đầu ranking. |
| k-NN — k nearest neighbors | Dự đoán từ k hàng xóm gần nhất. k=3 nghĩa là ba hàng xóm, không phải ba class. |
| Majority vote | Bỏ phiếu theo lớp: lớp có nhiều hàng xóm nhất trong top-k được chọn. |
| Tie / tie-break | Hòa / quy tắc phá hòa. Có thể hòa score hoặc hòa số phiếu; đây là hai trường hợp riêng. |
| Stable ranking / stable sort | Khi score bằng nhau, giữ thứ tự ban đầu của gallery. Quy tắc này làm kết quả có thể tái tạo. |
| Nearest tied class | Khi các lớp hòa số phiếu, chọn lớp có hàng xóm xuất hiện sớm nhất trong ranking trong số lớp hòa. |
| Prediction | Nhãn mà phương pháp dự đoán cho query. |
| ACC / accuracy | Tỷ lệ query được dự đoán đúng nhãn. Đúng 8/10 query thì ACC=80%. |
| ACC@k | Accuracy của bộ phân loại dùng k-NN. Trong repo này, **ACC@3 không có nghĩa nhãn đúng xuất hiện trong top-3**; nó là accuracy sau bỏ phiếu ba hàng xóm. |
| Top-k | k phần tử đầu trong ranking. Cần xem dùng top-k để bỏ phiếu, retrieval hay đo một chỉ số khác. |
| Relevant | Mẫu được coi là liên quan. Với MAP trong repo, gallery sample có cùng class với query là relevant. |
| Precision tại hạng r | Số relevant trong r phần tử đầu chia cho r. Ví dụ top-5 có ba mẫu đúng lớp thì precision tại hạng 5 là 3/5=60%. Khác numerical precision ở mục 4. |
| AP — average precision | Chất lượng ranking của một query: lấy precision ở mỗi hạng có relevant, cộng lại và chia cho tổng số relevant trong gallery. Relevant ở càng sớm thường càng tốt. |
| MAP — mean average precision | Trung bình AP của mọi query. Đánh giá toàn ranking, khác ACC chỉ chấm nhãn dự đoán. |
| Full-gallery MAP | AP/MAP tính với toàn gallery, không chỉ top-k. Đây là cách repo dùng. |
| NN agreement | Tỷ lệ query mà hai implementation chọn cùng gallery index gần nhất. Agreement 100% chỉ nói hai implementation đồng ý, không nói chúng phân loại đúng 100%. |
| NN-index disagreement | Hai implementation chọn khác sample gần nhất. Hai sample đó có thể cùng class, nên disagreement không nhất thiết làm ACC khác. |
| Ranking agreement | Hai ranking trùng nhau. Mạnh hơn chỉ NN agreement, vì so toàn bộ thứ tự chứ không chỉ vị trí đầu. |
| DTW — dynamic time warping | Cách so chuỗi bằng việc tìm đường căn chỉnh giữa các vị trí, cho phép hai chuỗi diễn ra với tốc độ khác nhau. Không phải một thuật ngữ OT. |
| LDTW trong repo | Score DTW chia cho độ dài chuỗi x. Đây là định nghĩa hiện dùng trong code; không tự đồng nhất tên này với những biến thể LDTW khác ngoài repo. |
| NDTW trong repo | Score DTW chia cho số bước trên đường căn chỉnh DTW được chọn. |
| Soft-DTW / gamma | Phiên bản làm trơn phép chọn nhỏ nhất trong DTW / tham số điều khiển mức làm trơn. Repo dùng raw score, không tự thay bằng một dạng đã hiệu chỉnh khác. |
| DTW window | Giới hạn vùng được phép căn chỉnh. “Không window” nghĩa là không đặt thêm giới hạn loại này. |

Ví dụ: gallery có A1, A2, B1, B2; query có nhãn A và ranking là B1, A1, A2, B2. ACC@1 sai, ACC@3 đúng vì hai phiếu A thắng một phiếu B. AP=(1/2+2/3)/2≈58.33%. Nếu một query khác có AP=100% thì MAP của hai query là khoảng79.17%. Hai implementation cùng chọn B1 thì NN agreement vẫn đạt100% trên query đó dù dự đoán sai.

## 3. Thiết kế thí nghiệm và lựa chọn tham số

| Thuật ngữ | Giải thích |
|---|---|
| Experiment / run / job / case | Experiment là thí nghiệm; run là một lần thực thi; job là đơn vị công việc runner quản lý; case là một trường hợp input/cấu hình. Một experiment có thể có nhiều run, mỗi run nhiều job/case. |
| Setting / configuration / config | Bộ lựa chọn cụ thể: dataset, kích thước, tham số, precision, số vòng, backend… |
| Parameter / hyperparameter | Tham số nói chung / tham số do người chạy chọn để điều khiển phương pháp. “Tuning parameters” trong báo cáo chủ yếu là chọn hyperparameter. |
| Default / preset | Giá trị mặc định trong code / bộ cấu hình đặt sẵn để làm mốc. Hai từ có thể trùng giá trị trong một experiment nhưng không luôn đồng nghĩa. |
| Selected / tuned | Bộ tham số đã được chọn / đã qua quá trình tuning. Không có nghĩa tối ưu trên mọi dữ liệu. |
| Profile | Trong bảng preset/selected, là một nhóm cấu hình tham số. Trong “profiling GPU”, lại mang nghĩa đo chi tiết hiệu năng, xem mục 7. |
| Tuning / search / retune | Thử các cấu hình để chọn bộ tốt theo tiêu chí / quá trình tìm kiếm / chọn lại tham số. |
| Candidate | Một bộ tham số ứng viên. Sáu candidate nghĩa là thử sáu bộ, không nhất thiết sáu giá trị của mỗi tham số. |
| Grid / grid search | Danh sách các bộ tham số cần thử / thử các bộ đó theo quy tắc. |
| Joint grid | Mỗi candidate quy định đồng thời nhiều tham số. Sáu cặp (a,b) là sáu candidate. |
| Cartesian grid | Thử mọi tổ hợp giữa các danh sách tham số. Sáu giá trị a và sáu giá trị b tạo 36 tổ hợp. |
| Budget | Ngân sách chạy: có thể là số candidate, thời gian, bộ nhớ hoặc số vòng. Cần xem budget đang tính theo đơn vị nào. |
| Selection rule | Quy tắc chọn candidate. Repo ưu tiên mean ACC@1, dùng MAP phá hòa, rồi thứ tự khai báo. |
| Eligibility / bị loại khỏi selection | Điều kiện để candidate được phép chọn. Candidate không đạt tiêu chí hội tụ tại cap có thể bị loại, dù vẫn giữ số liệu của nó. |
| Freeze / frozen parameters | Đóng băng bộ tham số và quy tắc sau tuning để dùng cho bước đánh giá. Không chọn lại theo kết quả TEST. |
| Frozen numerical sources | Các file code tính toán được giữ cố định bằng source hashes trong một run. Đổi file có thể làm checkpoint không còn tương thích. |
| Protocol / policy | Quy trình thống nhất / bộ quy tắc cụ thể. Ví dụ policy dừng quy định ngưỡng, khoảng kiểm tra và số vòng tối đa. |
| Benchmark | Bài đo có thiết kế để đánh giá tốc độ, bộ nhớ hoặc khả năng mở rộng. Có benchmark không đồng nghĩa đã chứng minh correctness. |
| Baseline | Phương pháp đối chứng để so sánh. Có thể là phương pháp cũ, thư viện bên ngoài hoặc bản dense tham chiếu. |
| Control / controlled comparison | Đối chứng để khảo sát một yếu tố / phép so giữ các yếu tố còn lại giống nhau. |
| Same-GPU benchmark | Các phương pháp được đo trên cùng GPU. Vẫn cần thống nhất precision, input, cách đo và tải máy để có phép so có ý nghĩa. |
| Ablation | Thay hoặc bỏ một yếu tố để xem ảnh hưởng. Ví dụ kernel generic so với specialized trên cùng input. |
| Pilot | Chạy thử phạm vi nhỏ để kiểm tra thiết kế, phát hiện lỗi và ước lượng khối lượng công việc. Không thay cho full evaluation. |
| Smoke test | Kiểm tra rất nhỏ xem chương trình có chạy được qua các bước chính không. |
| Toy / tiny fixture | Dữ liệu nhỏ, dễ kiểm tra, dùng thử correctness hoặc pipeline. Accuracy trên nó thường không đại diện dữ liệu thực. |
| Full / partial run | Đã chạy toàn khối lượng quy định / chỉ chạy một phần. Full phải xét trong phạm vi workload đã định nghĩa. |
| Panel | Một ô biểu đồ hoặc một nhóm phép đo. Tám panel synthetic không đồng nghĩa toàn paper chỉ có tám experiment. |
| Sweep | Quét nhiều giá trị của một biến, như n hoặc d. |
| Scaling / scalability | Cách thời gian/bộ nhớ thay đổi khi input lớn lên / khả năng xử lý input lớn. |
| Workload | Tổng công việc cần xử lý: số input, số cặp, số vòng và số cấu hình. |
| Deduplicate | Gộp các trường hợp trùng nhau để tránh tính lại. Hai profile có cấu hình giống nhau có thể dùng chung kết quả. |
| Reuse | Dùng lại artifact đã có thay vì giải lại. Cần giữ nguồn gốc; không gọi reuse là một lượt GPU solve mới. |
| Reproduce / reproduction | Tái hiện kết quả hoặc phương pháp với phạm vi phải được nêu rõ. Tái hiện công thức, output và runtime là các mục tiêu khác nhau. |
| Reimplementation / implementation | Tự viết lại phương pháp / một bản code cụ thể thực hiện phương pháp. Bản tự triển khai chưa mặc nhiên tương đương code tác giả. |
| Global optimum | Cấu hình tốt nhất trên toàn không gian tìm kiếm. Thử một grid nhỏ không chứng minh tìm được global optimum. |

## 4. Số học máy tính, FP và phép kiểm tra sai số

| Thuật ngữ | Giải thích |
|---|---|
| FP — floating point | Số thực dấu phẩy động: cách máy tính lưu số gần đúng bằng các bit. Trong báo cáo này FP không có nghĩa “false positive”. |
| FP32 / single precision | Kiểu số dấu phẩy động 32 bit. Thường có khoảng7 chữ số thập phân có nghĩa; không phải chính xác7 số sau dấu phẩy cho mọi giá trị. |
| FP64 / double precision | Kiểu 64 bit, thường khoảng15–16 chữ số có nghĩa. Hay dùng làm reference vì sai số làm tròn nhỏ hơn FP32; vẫn không phải số học chính xác tuyệt đối. |
| TF32 | Chế độ tính toán trên Tensor Cores: đầu vào có độ chi tiết phần trị thấp hơn FP32 đầy đủ, thường tích lũy theo FP32. Tensor lưu trữ vẫn có thể là FP32; TF32 không phải đồng nghĩa FP16. |
| IEEE FP32 / strict FP32 | Trong runner, yêu cầu phép tính FP32 đầy đủ thay vì đường dot-product TF32. Không bảo đảm các implementation có thứ tự cộng giống nhau hoặc output giống từng bit. |
| Numerical precision | Độ chi tiết của cách biểu diễn/tính số. Khác precision trong retrieval. |
| Dtype | Kiểu dữ liệu của tensor, ví dụ float32 hoặc float64. |
| Quantize / làm tròn input về FP32 | Chuyển input sang các giá trị biểu diễn được bằng FP32. Chuyển những giá trị đã làm tròn đó sang FP64 không phục hồi chữ số đã mất. |
| Floating-point rounding | Làm tròn vì một số không thể được lưu chính xác trong kiểu FP. |
| Non-associativity | Phép cộng FP có thể cho (a+b)+c khác a+(b+c). Đổi thứ tự reduction trên GPU có thể gây chênh nhỏ. |
| Finite / nonfinite | Hữu hạn / không hữu hạn. Inf và NaN là nonfinite; finite chỉ là điều kiện tối thiểu, chưa chứng minh đáp án đúng. |
| NaN / Inf | Not a Number: kết quả số không hợp lệ / vô cực. Có thể xuất hiện do phép tính không hợp lệ hoặc vượt khả năng biểu diễn. |
| Overflow | Giá trị quá lớn so với phạm vi kiểu số, có thể thành Inf. |
| Numerical stability | Mức độ thuật toán kiểm soát việc khuếch đại sai số. Nhiều bit hơn không tự sửa được mọi vấn đề ổn định. |
| Absolute error / max absolute error | Độ lớn chênh lệch giữa output và reference / giá trị lớn nhất trong các phần tử. Ví dụ1.0002 so với1.0000 có absolute error0.0002. |
| Relative error | Sai số được chia theo độ lớn reference. Cùng absolute error có thể nhỏ với reference lớn nhưng lớn với reference gần0. |
| Relative L2 error | Độ dài vector sai số chia cho độ dài vector reference: norm(output−reference)/norm(reference). Với matrix có thể xét tương đương vector hóa/Frobenius norm. |
| Norm / L2 / Frobenius | Cách đo độ lớn của vector hoặc matrix. L2 vector là căn tổng bình phương; Frobenius matrix là căn tổng bình phương mọi phần tử. |
| rtol / atol | Relative tolerance / absolute tolerance: sai số tương đối/tuyệt đối cho phép trong kiểm tra gần bằng. Quy tắc thông dụng: abs(output−reference)≤atol+rtol×abs(reference). |
| Parity | Hai implementation cho output đủ gần theo tiêu chí đã đặt. “Parity pass” không nhất thiết giống từng bit và chỉ áp dụng các case/đại lượng đã kiểm tra. |
| Score drift | Score thay đổi khi tăng số vòng so với checkpoint được chọn làm mốc. Drift0 ở chính checkpoint mốc là theo định nghĩa. |
| Raw score difference | Chênh lệch hai score chưa chuẩn hóa/hiệu chỉnh. Nếu chúng có định nghĩa khác nhau thì chênh này không phải lỗi parity. |
| Threshold | Ngưỡng dùng cho quyết định pass/fail hoặc dừng. |
| 1e-5 / 10^-5 | Ký hiệu khoa học:0.00001. 1e-3=0.001; 6.985e-9=0.000000006985. |

## 5. Thuật toán số và các thuật ngữ toán học tổng quát

| Thuật ngữ | Giải thích |
|---|---|
| Vector / matrix / tensor | Danh sách số một chiều / bảng số hai chiều / mảng số có thể nhiều chiều. Trong PyTorch, “tensor” còn là kiểu đối tượng dữ liệu kể cả vector và matrix. |
| Shape | Kích thước theo từng chiều của tensor. Shape2050×200 nghĩa là2050 hàng,200 cột. |
| Square / rectangular | Matrix vuông / chữ nhật. Hai độ dài bằng nhau tạo square; khác nhau tạo rectangular. |
| Squared Euclidean distance | Tổng bình phương chênh lệch từng tọa độ. Ví dụ(1,2) và(4,6):3²+4²=25. Euclidean distance thông thường là căn của25, tức5. |
| Dot product | Tích vô hướng: nhân từng cặp phần tử rồi cộng. Là phép tính cơ bản của nhiều kernel. |
| Affine / linear | Affine là phép tuyến tính cộng hằng số; linear không có phần hằng độc lập. Trong báo cáo cần đọc theo công thức cụ thể. |
| Taylor approximation | Xấp xỉ một hàm quanh một điểm bằng các đạo hàm. Nó có thể rất gần trong một vùng và kém hơn ở vùng khác. |
| Exact | Theo đúng biểu thức đang xét, thay vì xấp xỉ Taylor. Code FP dùng biểu thức “exact” vẫn chịu sai số làm tròn. |
| Loss / objective | Đại lượng cần tính hoặc tối ưu. Loss dùng tối ưu không nhất thiết chính là score dùng ranking. |
| Forward | Bước tính output/loss từ input. Trong benchmark này gồm việc giải theo số vòng quy định và tính cost. |
| Backward | Bước tính đạo hàm/gradient. F+B là tổng forward và backward, không phải thời gian backward riêng. |
| Gradient | Vector đạo hàm bậc một: cho biết giá trị hàm thay đổi theo từng thành phần input. |
| Analytic gradient / backward | Gradient từ công thức đạo hàm đã suy ra, thay vì lưu rồi đạo hàm qua mọi bước của quá trình lặp. |
| Autodiff / differentiation graph | Tự động tính đạo hàm / đồ thị các phép tính để hệ thống suy ra đạo hàm. |
| Hessian | Matrix đạo hàm bậc hai, mô tả sự thay đổi của gradient hay độ cong của hàm. |
| HVP — Hessian-vector product | Nhân Hessian với một vector. Có thể tính kết quả mà không tạo toàn matrix Hessian rất lớn. |
| First-order / second-order | Dùng thông tin đạo hàm bậc một / bậc hai. |
| Solver / solve | Thuật toán giải bài toán / thực hiện việc giải. Không nhất thiết trả nghiệm chính xác khi chỉ chạy hữu hạn vòng. |
| Iteration / fixed iterations | Một vòng cập nhật / chạy đúng ngân sách vòng đã đặt thay vì dừng khi đạt ngưỡng. Khác measured repeats: lặp nội bộ thuật toán so với lặp phép đo. |
| Schedule | Thứ tự/cách tổ chức cập nhật trong thuật toán. Trong báo cáo có cập nhật luân phiên và cập nhật đối xứng. |
| Residual | Đại lượng đo phần điều kiện còn chưa thỏa. Ý nghĩa/cách chuẩn hóa phụ thuộc solver; residual trong CG và residual r trong OPW không tự là cùng một đại lượng. |
| Convergence / hội tụ | Quá trình tiến gần nghiệm hoặc thỏa tiêu chí dừng đã chọn. Ranking ổn định không tự chứng minh mọi điều kiện số học đã hội tụ. |
| Tolerance / tau / ngưỡng dừng | Mức residual cho phép. Trong protocol OPW, tau=0.001; đây không phải tự động là sai số score0.001. |
| Residual stop / adaptive | Dừng khi kiểm tra residual đạt ngưỡng, nên các input có thể dùng số vòng khác nhau. “Adaptive” ở đây không phải tự tuning tham số theo TEST. |
| Check every50 | Chỉ kiểm tra tiêu chí dừng mỗi50 vòng. Vì vậy checkpoint đầu đạt ngưỡng chưa phải số vòng tối thiểu chính xác. |
| Cap / capped pair | Số vòng tối đa / cặp chạm giới hạn vòng nhưng vẫn chưa đạt ngưỡng. |
| Early stopping | Dừng trước ngân sách tối đa khi đủ điều kiện. |
| CG — conjugate gradient | Phương pháp lặp giải hệ phương trình tuyến tính thuộc lớp thích hợp, dùng các phép nhân matrix-vector. Trong benchmark HVP đặt ngân sách50 bước. |
| Damping | Thêm một thành phần điều chỉnh vào hệ cần giải để dễ ổn định hơn. Nó có thể làm thay đổi bài toán số đang giải, không chỉ thay tốc độ. |
| Schur complement | Cách biến đổi một hệ phương trình theo các khối để giải một hệ nhỏ hơn. Xuất hiện trong routine HVP. |
| Guarded fixed-step CG | CG có ngân sách bước cố định và kiểm tra tránh các phép tính hỏng như chia cho đại lượng không hợp lệ. Không nên hiểu là không có bất kỳ điều kiện bảo vệ nào. |
| FP64 oracle | Bản tham chiếu dùng FP64, thường đơn giản/dễ kiểm tra hơn, dùng so output. “Oracle” là công cụ đối chứng, không phải đáp án chân lý được bảo đảm. |
| LP / HiGHS | Linear programming: tối ưu hàm tuyến tính với ràng buộc tuyến tính / bộ solver dùng cho bài toán loại này. |
| Log-domain / LSE | Tính trong miền log để kiểm soát phạm vi số / log-sum-exp, phép tính log của tổng các số mũ theo cách ổn định hơn. Đây là kỹ thuật số học chung. |
| Reduction | Gộp nhiều phần tử thành ít hơn bằng tổng, max hoặc phép khác. Thứ tự gộp có thể ảnh hưởng sai số FP. |

## 6. Phần cứng và bộ nhớ

| Thuật ngữ | Giải thích |
|---|---|
| CPU / GPU | Bộ xử lý trung tâm / bộ xử lý có nhiều đơn vị tính song song, thường dùng tăng tốc các phép toán tensor. |
| GPU physical index | Số hiệu card vật lý mà server hiển thị. GPU vật lý1 có thể trở thành cuda:0 bên trong process sau khi lọc card. |
| CUDA_VISIBLE_DEVICES | Biến môi trường giới hạn các GPU process được thấy và ánh xạ lại số hiệu trong process. |
| Thread | Luồng thực thi. “Hai CPU thread” giới hạn mức song song CPU được yêu cầu ở phần cấu hình liên quan, không có nghĩa GPU chỉ dùng hai luồng. |
| Process / PID / child process | Tiến trình đang chạy / số định danh tiến trình / tiến trình con được tiến trình khác tạo ra. Hai cửa sổ terminal có thể chạy hai process riêng. |
| CPU RAM / GPU VRAM | Bộ nhớ chính dùng bởi CPU / bộ nhớ trên card GPU. Không cộng VRAM các card thành một vùng duy nhất cho một job thông thường. |
| HBM / SRAM | Bộ nhớ ngoài chip băng thông cao / bộ nhớ nhỏ, nhanh ở gần các đơn vị tính. HBM là tên một công nghệ; không phải mọi GPU đều dùng HBM. |
| CUDA | Nền tảng phần mềm để chạy tính toán trên GPU NVIDIA. |
| Driver | Phần mềm hệ thống điều khiển GPU và cung cấp khả năng tương tác cho chương trình. |
| CUDA Toolkit / runtime | Bộ công cụ phát triển CUDA / thành phần phục vụ chạy chương trình CUDA. Không tự đồng nhất với phiên bản CUDA được driver báo trên nvidia-smi. |
| cu128 / +cpu | Trong tên bản PyTorch, chỉ build CUDA12.8 / build CPU. Đây là thông tin package, không đủ xác nhận toàn môi trường của một run. |
| nvidia-smi | Công cụ đọc trạng thái card NVIDIA: bộ nhớ, utilization, nhiệt độ, process… |
| GPU-Util / utilization | Tỷ lệ thời gian GPU bận trong khoảng lấy mẫu. 0% ở một thời điểm không chứng minh card trống: process vẫn có thể giữ nhiều VRAM. |
| Allocation / allocated memory | Bộ nhớ được cấp cho dữ liệu. PyTorch allocated là phần allocator theo dõi cho tensor/đối tượng còn dùng, không phải tổng bộ nhớ mà toàn GPU đang dùng. |
| Peak memory | Mức bộ nhớ lớn nhất quan sát được trong đoạn đo. |
| Peak PyTorch allocated | Đỉnh bộ nhớ do PyTorch cấp, gồm cả live tensors đã tồn tại lúc reset peak. Runner FlashSinkhorn không trừ baseline allocation. |
| Extra allocation | Phần tăng thêm so với một mức nền đã đo/trừ. Không dùng tên này cho peak tuyệt đối nếu code không trừ nền. |
| Live tensors | Tensor còn được giữ và chưa giải phóng. Input có thể là live tensor trong suốt phép đo. |
| Reserved memory | Bộ nhớ allocator giữ để tái sử dụng, có thể lớn hơn allocated. Khác tổng VRAM trong nvidia-smi. |
| Memory fraction | Giới hạn tỷ lệ bộ nhớ đặt cho allocator liên quan. Giá trị0.45 không bảo đảm GPU-Util45%, không độc quyền card và không tự khống chế mọi framework khác. |
| Working budget / memory estimate | Ngân sách cho vùng dữ liệu làm việc / ước tính bộ nhớ trước khi chạy. Estimate không phải peak thực đo. |
| MB / MiB / GB | MB=10^6 byte; MiB=2^20 byte; GB=10^9 byte. Đổi đơn vị trước khi so, không coi MB và MiB bằng nhau. |
| OOM — out of memory | Không đủ bộ nhớ cho phép cấp phát. Khác skip trước khi thực sự cấp phát. |
| Shared load / shared resources | Các job cùng dùng CPU, RAM, disk hoặc GPU, có thể ảnh hưởng timing. Hai job ở GPU riêng vẫn có thể tranh CPU/I/O. |
| I/O | Đọc/ghi dữ liệu, ví dụ disk hoặc trao đổi qua hệ thống bộ nhớ. Trong “IO-aware”, cần xem cấp bộ nhớ nào đang được nói tới. |
| Data transfer | Chuyển dữ liệu giữa các vùng/thiết bị, thường CPU↔GPU. |

## 7. Cách tổ chức code và đo hiệu năng

| Thuật ngữ | Giải thích |
|---|---|
| Backend | Cách thực thi bên dưới cùng API: CPU, Torch CUDA, Triton, KeOps, JAX… Có thể cùng công thức nhưng tốc độ/sai số khác. |
| Kernel / GPU kernel | Một routine tính toán chạy trên GPU. Một experiment thường gọi nhiều kernel. |
| Dense / tensorized | Tạo và xử lý matrix đầy đủ. Dễ đối chiếu hơn nhưng có thể tốn nhiều bộ nhớ. |
| Streaming / online | Tính từng phần mà không giữ toàn matrix tương tác. “Online” ở đây không có nghĩa cần kết nối Internet. |
| Tiled / tile | Chia matrix thành các khối nhỏ / một khối như64×128. Mỗi lần kernel xử lý một phần dữ liệu. |
| Block / block-m / block-n | Kích thước hoặc nhóm công việc trong kernel; block-m/n trong lệnh là tham số tile của implementation. |
| Generic / specialized kernel | Kernel tổng quát / kernel được tối ưu cho một loại phép tính hoặc input cụ thể. Specialized chưa tự bảo đảm nhanh hơn ở mọi shape. |
| Fused / fusion | Ghép nhiều bước vào một kernel để giảm lần gọi và trao đổi dữ liệu trung gian. |
| Autotuning / bounded autotuning | Tự thử các cấu hình kernel để chọn cấu hình nhanh / chỉ thử trong tập ứng viên và giới hạn tài nguyên đã đặt. |
| Warp / stages / register / spill | Nhóm luồng GPU cùng thực thi / các giai đoạn pipeline / vùng lưu trữ rất nhanh cho luồng / dữ liệu phải chuyển ra bộ nhớ khác khi thiếu register. Đây là thông số tài nguyên của kernel. |
| API | Giao diện gọi code: tên hàm, đối số và loại output. Cùng API không bảo đảm cùng implementation. |
| Adapter / wrapper | Lớp nối để tương thích hai giao diện / code bao quanh một routine hoặc lệnh để cấu hình cách chạy. Adapter có thể thay hành vi nếu nó đổi solver bên dưới. |
| Entry point / runner / script / module | Điểm bắt đầu chạy / code điều phối thí nghiệm / file lệnh / đơn vị code Python có thể import hoặc chạy bằng python -m. |
| Pipeline / end-to-end | Chuỗi các bước xử lý / đo trọn chuỗi đã quy định. Pipeline có thể gồm load, transfer, tính distances, ranking và evaluation. |
| Pair-by-pair / batching / batch | Xử lý từng cặp / gom nhiều cặp để xử lý / một nhóm input được xử lý cùng lúc. Batch16 không nhất thiết là16 query; cần đọc runner. |
| Chunk | Phần workload để xử lý/lưu checkpoint. “Chunk4 query” nghĩa là một phần gồm4 query; khác batch16 cặp. |
| Precompute | Tính trước rồi tái sử dụng. Khi loại bước này khỏi timing, số đo không bao gồm toàn chi phí tạo dữ liệu trung gian. |
| Setup | Chuẩn bị trước phép đo, như tạo input và giải những trạng thái được cache. |
| Cached / cache | Đã lưu / vùng lưu kết quả hoặc code biên dịch để dùng lại. Có cache khiến lần sau có thể nhanh hơn lần đầu. |
| Warmup | Chạy trước để biên dịch, khởi tạo và làm nóng cache. Không lấy các lần này vào thống kê timing chính. |
| Repeat / measured repetition | Chạy lặp phép đo. 50 repeats là50 mẫu timing; khác10 iterations bên trong mỗi lần chạy. |
| Raw samples | Từng số đo gốc trước khi tính mean/median. Chỉ có mean thì không tính lại độ phân tán được. |
| Timer | Cơ chế đo thời gian. |
| CUDA events | Mốc thời gian ghi trên luồng GPU để đo đoạn công việc GPU. Không tự bao gồm toàn chi phí CPU hay disk. |
| Synchronization / block_until_ready | Đợi tác vụ hoàn tất trước khi đọc output/thời gian. Nếu không đợi, có thể chỉ đo thời gian gửi lệnh. |
| Wall-clock time | Thời gian thực trôi qua giữa hai mốc trên CPU. Chỉ gồm những bước được đặt trong đoạn đo. |
| Runtime / latency | Thời gian chạy / thời gian hoàn thành một yêu cầu hoặc thao tác được định nghĩa. Cần xem tính trên một pair, matrix hay cả pipeline. |
| Speedup | Thời gian baseline chia thời gian phương pháp được so. 2× nghĩa là phương pháp dùng khoảng nửa thời gian;0.8× nghĩa là phương pháp chậm hơn baseline. |
| Crossover / rank reversal | Điểm hai phương pháp đổi thứ tự nhanh/chậm / thứ hạng đảo giữa hai phép so. Không tự cho biết nguyên nhân. |
| Profiler / profiling / trace | Công cụ đo chi tiết / việc đo / bản ghi timeline các tác vụ. Dùng phân biệt chi phí kernel, CPU và synchronization. |
| Kernel launch / launch overhead | Gửi kernel tới GPU / chi phí tổ chức và gửi lần gọi, ngoài phần số học chính của kernel. |
| Compiler / compile | Chương trình dịch code sang dạng thực thi / thực hiện việc dịch. |
| JIT — just-in-time compilation | Biên dịch khi chương trình chạy hoặc gặp input mới. Lần đầu có thể chậm hơn vì gồm biên dịch. |
| Interpreter / offline compiler | Chạy theo cơ chế diễn giải / biên dịch kiểm tra mà chưa chạy trên GPU. Pass không chứng minh timing hoặc correctness GPU thực. |
| Upper bound | Giới hạn trên. Tile upper bounds không có nghĩa mọi case chắc chắn chọn đúng tile đó. |
| Timeout / deadline / OOT / TO | Hết thời gian cho phép / hạn thời gian / out of time / ký hiệu timeout trong bảng. Deadline của child process có thể gồm startup, warmup và mọi repeats, không phải latency một phép toán. |
| ms / s | Millisecond / giây.1000ms=1s. |

## 8. Kiểm chứng, audit và mức độ bằng chứng

| Thuật ngữ | Giải thích |
|---|---|
| Correctness | Tính đúng của implementation theo công thức và điều kiện đã định nghĩa. Một benchmark nhanh chưa chứng minh correctness. |
| Test / pass / fail / skip | Trường hợp kiểm tra / đạt tiêu chí / không đạt / không thực hiện. Skip trên máy CPU không phải GPU pass. |
| Warning | Cảnh báo khi chạy, không tự là failure. Vẫn cần đọc nguyên nhân cụ thể. |
| Regression test | Kiểm tra nhằm phát hiện một lỗi cũ tái xuất hiện hoặc thay đổi phá hành vi đã biết. |
| Integration test | Kiểm tra nhiều thành phần nối với nhau đúng, ví dụ đọc tham số→tính matrix→đánh giá→lưu file. |
| Functional check | Kiểm tra chương trình thực hiện được chức năng dự kiến. Chưa phải performance measurement được kiểm soát. |
| Diagnostics | Các chỉ số kiểm tra bổ sung: output hữu hạn, sai số, residual, số vòng… Có thể tính ngoài đoạn timing. |
| Audit / auditor | Kiểm tra lại kết quả từ artifact / chương trình hoặc người làm bước đó. Trong repo, có thể đọc matrix rồi tính lại MAP/ACC, kiểm hash và metadata. Không phải chứng nhận độc lập của một tổ chức bên ngoài. |
| Artifact | File/sản phẩm do run tạo ra: matrix, checkpoint, log, environment, samples, summary… Chỉ là một ảnh chụp console thì bằng chứng ít hơn bộ artifact đầy đủ. |
| Audited artifact | Artifact đã được kiểm tra theo phạm vi được nêu. Audit một số score mẫu không có nghĩa audit toàn bộ output hay kernel. |
| Sampled verification | Kiểm chứng trên một tập con. Ví dụ36 score không thay cho kiểm toàn bộ hàng triệu pair. |
| Independent reference | Bản tính tham chiếu được viết theo cách khác để giảm khả năng cùng lỗi implementation. Vẫn có thể chia sẻ một diễn giải công thức sai; không bảo đảm tuyệt đối. |
| Supplied log / console log | Nhật ký người dùng cung cấp / chữ in ra terminal. Có thể thiếu source revision, môi trường và raw samples. |
| Metadata / environment | Thông tin về run / cấu hình môi trường: GPU, package versions, lệnh, seed, source hashes… Không phải score chính. |
| Provenance | Nguồn gốc dữ liệu: run nào sinh số đo, dùng code nào, file nào đã parse. |
| Parse | Đọc dữ liệu theo cấu trúc để chuyển thành dạng dùng được, ví dụ trích console thành CSV. Parse đúng không tự xác nhận run đo đúng. |
| Canonical input / fingerprint | Dữ liệu sau khi đưa về một dạng chuẩn / dấu nhận diện nội dung đó, thường bằng hash. Hash archive và hash canonical input không phải cùng đối tượng. |
| Hash / SHA256 / source hash | Dấu kiểm tra tính từ nội dung / thuật toán hash cụ thể / hash file code. Hash khớp hỗ trợ xác nhận nội dung giống; không chứng minh code hay kết quả đúng. |
| Checkpoint | Trạng thái được lưu để kiểm tra hoặc tiếp tục. Không nhất thiết là weights của mạng neural; ở đây có matrix/chunk/số vòng và metadata. |
| Resume | Tiếp tục từ checkpoint, tránh tính lại phần đã hoàn tất. |
| Migration / resume migration | Chuyển dữ liệu/schema checkpoint sang dạng tương thích theo một quy tắc giới hạn. Không đồng nghĩa chạy lại các phép tính cũ. |
| Numerical failure / code failure | Không đạt tiêu chí số học / chương trình lỗi hoặc không thực hiện được. Timeout và skip cần ghi riêng. |
| Nonfinite output | Output chứa NaN/Inf. Có timing không làm output này trở thành đáp án hợp lệ. |
| SKIP / preflight skip | Bỏ qua / bỏ trước khi chạy vì kiểm tra điều kiện, như memory estimate vượt giới hạn. Không đổi thành OOM thực đo. |
| NR — not reported | Log không có số đo baseline đó. Không tự kết luận vì timeout hay sai số. |
| NC trong báo cáo | Paper không có số numeric tương ứng trong những nguồn bảng được dùng để đối chiếu. Không đồng nghĩa paper không nghiên cứu workload đó. |
| Running / completed | Trạng thái đang chạy / hoàn tất mà runner đã lưu. File ghi running có thể cũ nếu process đã dừng; completed cần xét phần việc nào đã hoàn tất. |

## 9. Thống kê và cách đọc số liệu

| Thuật ngữ | Giải thích |
|---|---|
| Mean | Trung bình cộng. Với timing1,2,9ms, mean=4ms. |
| Median | Trung vị sau khi sắp số. Cùng bộ1,2,9ms có median=2ms. Ít bị một số đo quá lớn kéo lên hơn mean. |
| Min / max | Giá trị nhỏ nhất / lớn nhất. Max residual có thể thuộc một pair, không phải residual điển hình của mọi pair. |
| SD — standard deviation | Độ lệch chuẩn: mức các giá trị phân tán quanh mean. SD lớn nghĩa các split/repeat khác nhau nhiều. |
| Sample SD | SD ước lượng từ mẫu, thường dùng mẫu số n−1. Không phải sai số chuẩn của mean. |
| Mean (SD) | “79.762 (14.434)” là mean79.762% và sample SD14.434 điểm phần trăm trên các split được nêu; không phải79.762±14.434% CI95%. |
| Dispersion / variability | Độ phân tán / mức biến động giữa các lần đo, split hoặc input. |
| Percentage point / điểm phần trăm | Hiệu của hai tỷ lệ phần trăm:60% lên80% là tăng20 điểm phần trăm. Tăng tương đối là(80−60)/60≈33.33%. |
| Paired comparison / paired query | So hai phương pháp trên cùng query. Giữ mối tương ứng khi tính difference, thay vì coi là hai tập độc lập. |
| Difference / effect size | Chênh lệch đo được / độ lớn ảnh hưởng. Với MAP, có thể lấy AP_A−AP_B từng query rồi trung bình. |
| Bootstrap / resample | Lấy lại các query có hoàn lại nhiều lần để ước lượng biến động của đại lượng. Một query có thể xuất hiện nhiều lần trong một mẫu bootstrap. |
| Paired bootstrap | Khi lấy lại một query, lấy cả kết quả A và B của query đó để bảo toàn cặp so sánh. |
| Percentile bootstrap | Lấy các phân vị của phân phối bootstrap làm hai đầu khoảng. Ví dụ CI95% thường dùng phân vị2.5% và97.5%. |
| Percentile / phân vị | Ngưỡng mà một tỷ lệ quan sát nằm phía dưới. Median chính là phân vị50%. |
| CI — confidence interval | Khoảng tin cậy xây bằng một quy trình thống kê. CI95% mô tả mức bao phủ của quy trình qua nhiều mẫu giả định; không có nghĩa95% query được dự đoán đúng. |
| Conditional CI / CI có điều kiện | CI trong báo cáo giữ gallery và tham số đã chọn cố định, chỉ lấy lại query. Nó không đo biến động do chọn lại TRAIN, tuning hoặc đổi gallery. |
| McNemar test | Kiểm định so đúng/sai của hai phương pháp trên cùng query; tập trung query mà một phương pháp đúng và phương pháp kia sai. |
| Exact two-sided McNemar | Phiên bản exact dùng phân phối nhị thức dưới giả thuyết không khác biệt; two-sided xét khác biệt theo cả hai chiều. “Exact” không có nghĩa kết luận chắc chắn đúng. |
| p-value | Xác suất, dưới giả thuyết không khác biệt và các giả định kiểm định, quan sát mức chênh ít nhất cực đoan như đã thấy. Không phải xác suất “phương pháp A sai” hay “giả thuyết không khác biệt đúng”. |
| Multiple comparisons | Kiểm nhiều cặp/giả thuyết, làm tăng cơ hội thấy một khác biệt tình cờ nếu không điều chỉnh. |
| Holm adjustment | Cách điều chỉnh p-values khi kiểm nhiều giả thuyết để kiểm soát lỗi báo khác biệt trong cả nhóm kiểm định. Không tự biến mọi CI thành CI đã điều chỉnh. |
| Statistical superiority / significance | Lợi thế được hỗ trợ bởi phân tích thống kê / khác biệt vượt tiêu chí kiểm định đặt trước. Mean cao hơn một chút chưa tự đủ; significance cũng không tự nghĩa hiệu quả thực tế lớn. |
| Selection variability | Biến động do lấy TRAIN khác hoặc chọn tham số khác. Query bootstrap với tham số cố định chưa bao gồm phần này. |
| Log-log slope / fitted slope | Hệ số góc khi fit log(memory) theo log(n). Nếu memory≈c×n^p thì slope xấp xỉp. Đây là số fit trong miền dữ liệu đã đo, không phải chứng minh độ phức tạp tiệm cận. |
| Linear / quadratic scaling / O(n) / O(n²) | Tăng gần tỷ lệ kích thước / tăng gần bình phương kích thước / ký hiệu tốc độ tăng về mặt độ phức tạp. Gấp đôi n thì phần tuyến tính khoảng gấp đôi, phần bậc hai khoảng gấp bốn trong miền tương ứng. |
| Rounded mean | Mean đã làm tròn khi in console. Ratio tính từ nó có thể khác một chút so ratio từ full-precision raw samples. |
| Causal decomposition / phân rã nhân quả | Gán chênh lệch cho từng nguyên nhân. Thay lần lượt các yếu tố cho thấy tác động trên đường đối chứng đó, nhưng chưa chứng minh một phân rã duy nhất hoặc độc lập thứ tự. |

## 10. Repository, môi trường phần mềm và file

| Thuật ngữ | Giải thích |
|---|---|
| Local / server / workspace | Máy người dùng / máy chạy từ xa / thư mục dự án đang làm việc. Kiểm tra CPU local không thay kiểm tra GPU server. |
| WSL | Môi trường chạy Linux trong Windows. Dùng developer checks ở đây. |
| Conda environment | Môi trường Python có bộ package riêng. “conda activate minh” chọn môi trường tên minh, không phải tự đăng nhập một tài khoản GPU riêng. |
| Package / library / dependency | Gói phần mềm / thư viện / thành phần mà code cần để chạy. |
| PyTorch / Torch | Thư viện tensor và tính toán CPU/GPU. Torch là cách gọi ngắn trong báo cáo. |
| NumPy / SciPy | Thư viện mảng số / thư viện thuật toán khoa học, dùng cho CPU reference và kiểm chứng. |
| Numba | Công cụ biên dịch một số hàm Python số học, dùng tăng tốc CPU như DTW trong repo. |
| Triton | Ngôn ngữ/công cụ viết và biên dịch GPU kernels. Không phải tên một dataset. |
| KeOps / PyKeOps | Công cụ tính các phép toán trên tương tác cặp bằng biểu thức và reduction mà có thể tránh tạo toàn matrix. PyKeOps là giao diện Python. |
| GeomLoss | Thư viện dùng làm baseline trong benchmark; có backend tensorized và online/KeOps. |
| JAX / XLA | Thư viện mảng số và autodiff / hệ thống biên dịch, tối ưu các phép tính. |
| OTT-JAX / OTT-Hessian / Lineax | Bộ phần mềm bên ngoài dùng làm baseline / nguồn routine HVP / thư viện giải hệ tuyến tính. Trong repo, adapter CG đã thay một phần cách giải nên không gọi timing là raw Lineax. |
| Repository / repo | Kho mã nguồn và file dự án được quản lý bằng Git. |
| Git / branch / main | Hệ thống quản lý phiên bản / nhánh phát triển / tên nhánh chính trong repo này. “main (2).pdf” là tên tài liệu khác, không phải branch. |
| Commit / revision | Một phiên bản được lưu trong lịch sử Git. Chuỗi3813c08 là ID rút gọn; phải biết commit của run mới truy vết code sinh kết quả được. |
| Checkout / snapshot | Phiên bản đang lấy ra để làm việc / bản chụp file tại một thời điểm. Checkout hiện tại có thể khác phiên bản sinh artifact cũ. |
| Pin / pinned revision | Cố định đúng một phiên bản dependency/source để tránh vô tình đổi code khi chạy lại. |
| Upstream / author code | Nguồn dự án bên ngoài mà mình lấy code / code do tác giả phát hành. Adapter hoặc chỉnh sửa của repo cần ghi riêng. |
| Revert | Commit đảo lại một thay đổi trước đó. Không xóa mọi dấu vết khỏi lịch sử Git. |
| Archive | Bản lưu/gói file. Dataset archive, kết quả archive và Git history là các loại khác nhau. |
| Git ignore | Quy tắc không tự theo dõi một số file, như outputs. File local còn tồn tại nhưng chưa chắc có trên GitHub. |
| Schema | Cấu trúc file được quy định: tên trường, kiểu dữ liệu và quan hệ giữa các trường. Hai file tham số JSON có thể dùng schema không tương thích. |
| Override | Giá trị được truyền để thay mặc định, thường qua command-line hoặc biến môi trường. Chỉ đọc default trong code chưa xác nhận giá trị thực tế của run. |
| Environment variable | Biến môi trường như CUDA_VISIBLE_DEVICES hoặc FLASHOPW_AUTOTUNE, ảnh hưởng cách process chạy. |
| Source / source code | Nguồn file/code. “Paper source” trong bảng lại chỉ nơi trích số liệu, như Table18. |
| CSV | Bảng văn bản với hàng/cột, dễ đọc bằng spreadsheet hoặc Python. |
| JSON | File dữ liệu có các trường tên và giá trị, thường lưu cấu hình/metadata. |
| NPZ | File gói các mảng NumPy, dùng lưu distance matrices và labels. |
| PNG / figure / plot / overview | File ảnh / hình trong báo cáo / biểu đồ / hình tổng quan nhiều panel. |
| `.py` / `.sh` | File Python / script shell. Lệnh bash scripts/... chạy wrapper, python -m experiments... chạy module Python. |
| LaTeX / `.tex` | Hệ thống soạn tài liệu / file nguồn của báo cáo. |
| pdfLaTeX / XeLaTeX / LuaLaTeX | Các trình biên dịch LaTeX thành PDF. Kiểm tra cấu trúc file chưa đồng nghĩa đã biên dịch PDF thành công. |
| UTF-8 / font / `.bib` | Cách mã hóa chữ trong file / bộ chữ hiển thị / file dữ liệu tài liệu tham khảo của LaTeX. Báo cáo dùng bibliography trực tiếp nên không cần `.bib` riêng. |
| Eq. / Table / Figure / Appendix / caption | Phương trình / bảng / hình / phụ lục / chú thích của bảng hoặc hình. |
| Journal / paper / v3 | Bài báo tạp chí / bài báo nói chung / phiên bản3. Trong báo cáo, “journal” còn là nhãn phân biệt công thức từ OWD_journal.pdf. |

## 11. Những câu trong báo cáo được diễn giải bằng ngôn ngữ thường

| Câu rút gọn | Nghĩa thực tế |
|---|---|
| “TRAIN gallery28/query28, ba split” | Chọn28 mẫu TRAIN làm danh sách đối chiếu,28 mẫu TRAIN khác làm mẫu cần dự đoán; làm việc này theo ba cách chọn dữ liệu. |
| “CPU FP64 oracle, GPU FP32 parity” | Có bản tính CPU với độ chi tiết số cao hơn để so với bản GPU32 bit; kiểm tra chênh lệch trong ngưỡng, không yêu cầu giống từng bit. |
| “Artifact đã audit” | Đã mở những file kết quả được nêu và kiểm tra lại các phép tính/metadata theo phạm vi audit. Không tự nghĩa mọi kernel hoặc toàn paper đã được kiểm chứng. |
| “Fixed200 và residual stop” | Một chế độ luôn chạy200 vòng; chế độ kia chạy đến khi tiêu chí sai lệch đạt ngưỡng hoặc chạm số vòng tối đa. |
| “Tuned accuracy cao hơn preset” | Bộ tham số được chọn trên TRAIN cho tỷ lệ đúng cao hơn bộ đặt sẵn ở phép đánh giá đang xét. Không tự chứng minh tốt hơn trên mọi TEST. |
| “Mean79.762 (SD14.434) trên ba TRAIN split” | Accuracy trung bình của ba lần chia TRAIN là79.762%; các accuracy giữa split biến động khá lớn. Đây chưa phải accuracy full TEST hay CI. |
| “10 warmup,50 repeats,10 iterations” | Mỗi phép chạy có10 vòng thuật toán; bỏ10 lượt chạy làm nóng; sau đó lấy50 lượt đo thời gian. |
| “Speedup RTX/Paper=0.96/1.60” | Speedup đo trên RTX là0.96×, trong paper là1.60×. Dấu slash phân cách hai số, không yêu cầu lấy0.96 chia1.60. |
| “SKIP/OOM” | Phía reproduction bỏ chạy trước theo điều kiện; phía paper ghi không đủ bộ nhớ. Đây là hai trạng thái khác nhau. |
| “NR/OOT” | Phía reproduction không có số đo; phía paper ghi quá thời gian. Chưa kết luận server cũng bị timeout. |
| “NN agreement100% nhưng ACC62.5%” | Hai implementation luôn chọn cùng hàng xóm đầu, nhưng các dự đoán chỉ đúng nhãn ở62.5% query. |
| “Full TEST:2050×200×17” | Có2050 query, mỗi query so200 gallery sample, làm cho17 cấu hình khác nhau:6970000 pair computations theo workload quy định. |
| “Chunk297/8721” | Đã xử lý297 phần của tổng8721 phần workload được định nghĩa; không phải297 query hay297 thí nghiệm. |
| “Recorded job times không phải end-to-end latency” | Tổng các thời gian con được ghi không tự bằng thời gian thực từ lúc bắt đầu ứng dụng đến lúc có kết quả cuối. |

Để đọc báo cáo lần đầu, nên hiểu trước gallery/query, TRAIN/TEST, ACC/MAP, FP32/FP64, parity/audit, preset/selected, fixed iterations/residual stop và warmup/repeats. Sau đó tra các nhóm còn lại khi gặp từ chưa rõ.

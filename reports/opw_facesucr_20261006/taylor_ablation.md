# Taylor có giải thích khoảng giảm MAP/ACC không?

Ngày 06/10/2026, chạy các đối chứng CPU FP64 trên đúng 512 cặp trong
archive server, cùng features, uniform marginals, thứ tự cập nhật f rồi g,
và `(lambda1,lambda2,sigma)=(1,0.1,1)`. Không tìm tham số hoặc chọn k theo
test labels; đây là phân tích nguyên nhân trên pilot, không phải đánh giá
generalization của một phương pháp đã được tune.

**Trên pilot này, Taylor có ảnh hưởng nhỏ tới MAP trong preset main,
nhưng không phải nguyên nhân chính của khoảng cách với OPW journal.**

## Cost dùng cho các đối chứng

Đặt `F=(i/N-j/M)²`, `D=||x-y||²`, `q0=lambda2*log(sigma*sqrt(2*pi))-lambda1`.

- Inverse chính xác, bỏ hằng số: `D + lambda1*F/(1+F) + gamma*F`.
- Taylor/affine, bỏ hằng số: `D + lambda1*F + gamma*F`.
- Prior relative main: `gamma=lambda2/(2*sigma²)=0.05`.
- Prior journal: `gamma=lambda2/[2*sigma²*(1/N²+1/M²)]=429.025`
  khi `N=M=131`.

Hai cost đều có cùng hằng số `q0`, được bỏ trong solve. Sai số cost
Taylor trừ inverse là `lambda1*F²/(1+F)`; phép đối chứng vì vậy thay đúng
một thành phần khi giữ nguyên gamma.

Hai cách xếp hạng được kiểm tra: score từ thế `mean(f)+mean(g)-epsilon+q0`
theo cùng quy ước như main (19), và spatial score `<P,D>`. Score từ thế cho
cost inverse/journal là một đối chứng mở rộng theo cùng quy ước, không
gọi nó là score OPW journal trong paper. Journal xếp hạng bằng spatial score.

## Đổi từng yếu tố trên một đường đối chứng

| Order term | Prior | Score | Vòng | MAP (%) | ACC@1 (%) |
| --- | --- | --- | ---: | ---: | ---: |
| Taylor | Main relative | Thế f/g, main (19) | 200 | 68.449 | 62.50 |
| Inverse chính xác | Main relative | Cùng score thế f/g | 200 | 69.576 | 62.50 |
| Inverse chính xác | Journal | Cùng score thế f/g | 200 | 78.199 | 87.50 |
| Inverse chính xác | Journal | Spatial `<P,D>` | 200 | 83.375 | 87.50 |
| Inverse chính xác | Journal | Spatial `<P,D>` | 20 | 84.034 | 93.75 |

Các bước trên lần lượt thay Taylor, prior normalization, score và số vòng.
Chênh lệch MAP trên đường này: `+1.127`, `+8.623`, `+5.176`, `+0.659`
điểm phần trăm. Chênh lệch ACC@1: `0`, `+25`, `0`, `+6.25` điểm.
Đây là thay đổi có điều kiện trên đường đối chứng đã chọn; tương tác giữa
các yếu tố khiến chúng không phải một phân rã đóng góp duy nhất hoặc
áp dụng cho mọi dataset/parameter.

## Kiểm tra Taylor với prior journal giữ nguyên

200 vòng, spatial score:

| Order term | MAP (%) | ACC@1 (%) |
| --- | ---: | ---: |
| Inverse chính xác | 83.375 | 87.50 |
| Taylor | 83.375 | 87.50 |

Không chỉ chỉ số tổng hợp: toàn bộ gallery rankings của 16 query giống
nhau. Với score từ thế, rankings cũng giống nhau, MAP 78.199%, ACC@1
87.50% cho cả hai. Score không bằng chính xác: max difference `6.54e-6`
cho score thế và `2.42e-5` cho spatial score.

Với prior relative và spatial score, Taylor cho MAP **59.952%** so với
inverse **59.318%**, ACC@1 cùng **56.25%**. Như vậy Taylor không luôn
làm giảm MAP; hướng ảnh hưởng còn phụ thuộc score và prior. Đổi từ loss
main sang spatial score riêng lẻ cũng không cải thiện pilot main này.

## Kiểm chứng và giới hạn

- Nhánh affine-relative/main-score tái tạo archived affine dense 200
  vòng với max score error `2.22e-16`.
- Nhánh inverse-journal/spatial/20 vòng tái tạo archived OPW journal với
  max score error `2.76e-14`.
- Cost relative sau 200 vòng có max marginal L1 khoảng `8.4e-4`–`8.7e-4`.
  Prior journal mạnh hơn có max marginal L1 khoảng `0.0281` sau 200 vòng
  và `0.1049` sau 20 vòng. Các số trên là **kết quả finite-iteration**,
  không được gọi là optimum đã hội tụ.
- Tăng từ 20 lên 200 vòng ở journal làm ACC@1 của pilot giảm từ 93.75%
  xuống 87.50%; hội tụ solver và chất lượng phân loại không đồng nghĩa.
- Chỉ có 16 query. Không dùng những kết quả này để chọn tham số cuối
  cùng hoặc khẳng định Taylor vô hại trong mọi tình huống.

Lõi GPU không giải thích sai khác với archived affine dense: kiểm tra
trước đó cho max error `9.58e-8`, mọi ranking/prediction giống nhau.
Khoảng cách chất lượng đang thấy chủ yếu gắn với **mô hình/prior, score
và preset**, thay vì sai số tính toán của FlashOPW trên pilot này.

## Hướng xử lý vẫn ưu tiên main

Giữ công thức main làm mặc định. Không bỏ Taylor chỉ để tìm lại số của
journal. Với prior đã đổi đơn vị, cần chọn lại `sigma/lambda1/lambda2`
bằng validation trong training split; tham số journal không tự động
chuyển sang main với cùng độ mạnh temporal constraint.

Có thể nghiên cứu thêm biến thể affine giữ Gaussian normalization của
journal để tách mục tiêu tăng tốc OPW journal khỏi metric main được sửa.
Biến thể đó vẫn có cost bình phương mở rộng, nhưng phải đặt tên riêng và
không âm thầm đổi công thức main.

Script và dữ liệu phân tích local:

- `outputs/opw_server_review_20261006/taylor_ablation.py`
- `outputs/opw_server_review_20261006/taylor_ablation.csv`
- `outputs/opw_server_review_20261006/taylor_ablation.json`
- `outputs/opw_server_review_20261006/taylor_ablation_matrices.npz`

Lặp lại từ repo root:

```bash
python -c "import runpy; runpy.run_path('outputs/opw_server_review_20261006/taylor_ablation.py', run_name='__main__')"
```

Script SHA256 (chuẩn hóa CRLF):
`ed7eb45672a5fc4785dd3b6676c2db28b37014a671f196e4609b668a389107c1`.
Các file phân tích trong outputs chỉ có ở workspace local; không đổi
production solver hoặc archive server.

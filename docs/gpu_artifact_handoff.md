# Chuyển các file GPU đã có qua Git

Script chỉ thu thập kết quả có sẵn, không chạy CUDA hoặc chạy lại thí nghiệm.
Hai nguồn mặc định là `outputs/opw_group1_gpu_v1/` và
`outputs/paper_20261005T090633.880402Z/`. File `collection_manifest.json`
ghi danh sách tìm thấy/còn thiếu. Nếu tên thư mục thực tế khác, cần sửa
đường dẫn nguồn cho đúng trước khi thu thập; không coi file thiếu là đã được tạo.

Chạy khối sau trong terminal server. Các lệnh dừng khi có lỗi. Thư mục xuất
phải chưa tồn tại; lần sau dùng tên mới nếu muốn thu thập lại.

```bash
set -e
cd /home/doanpt/minh.nd/flash-opw
conda activate minh
git switch main
git pull --ff-only origin main
python scripts/collect_report_gpu_artifacts.py \
  --export-dir reports/gpu_artifacts_20261008
git add -f -- reports/gpu_artifacts_20261008/
git diff --cached --stat
git commit --only -m "Add existing GPU artifacts for experiment report audit" \
  -- reports/gpu_artifacts_20261008/
git push origin main
```

Chỉ thư mục xuất được đưa vào commit này; bản gốc trong `outputs/` được giữ
nguyên. `git add -f` cho phép đưa cả các file kết quả bên trong thư mục
`outputs` lồng vào bản xuất lên Git. Nếu server có thay đổi code chưa commit
hoặc đang ở nhánh khác, giải quyết trạng thái Git đó trước; không dùng
`reset --hard`, force push hoặc stash tự động.

Sau khi server push thành công, chạy trên PowerShell tại repo Windows:

```powershell
git pull --ff-only origin main
```

Kết quả sẽ ở `reports/gpu_artifacts_20261008/`. Thu thập file không thay
thế bước kiểm tra nội dung GPU/CPU, trạng thái hoàn tất và số liệu.

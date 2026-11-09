# SGU26_PPNCKH_Group

Đồ án môn Phương pháp nghiên cứu khoa học trong Khoa học dữ liệu: phân tích Ronao & Cho (2016) và tái hiện một phần CNN nhận dạng hoạt động con người trên UCI HAR.

## Bắt đầu

Đọc [protocol thực nghiệm](docs/research/protocol.md), sau đó mở [notebook chính](notebooks/01_har_experiment.ipynb). Notebook mặc định ở chế độ `inspect`, xác minh cache dữ liệu và đọc kết quả hiện có. Chọn `prepare` khi cần tạo dữ liệu lần đầu; chỉ khởi động huấn luyện khi đổi chế độ rõ ràng.

[Kế hoạch một tuần](docs/superpowers/plans/2026-11-09-har-research-project.md) và [nhật ký tiến độ](docs/research/progress.md) theo dõi các mốc. [Bảng dẫn chứng](docs/research/literature-matrix.md), [bảng đối chiếu](docs/research/original-vs-reproduction.md) và [bản thảo Chương 1–3](report/ban_thao_chuong_1_3.md) phục vụ viết báo cáo.

## Chạy trên Windows

Môi trường `.venv` trong workspace dùng Python 3.12 và đã có các thư viện cần thiết. Các lệnh dưới đây chạy từ thư mục gốc bằng PowerShell:

```powershell
# Kiểm tra các hành vi quan trọng của dữ liệu và mô hình
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Đọc tín hiệu, chia theo người và chuẩn hóa chỉ trên fit-train
.\.venv\Scripts\python.exe -m har_project prepare

# Chạy thử một epoch, kết quả nằm riêng trong results/smoke
.\.venv\Scripts\python.exe -m har_project train --models cnn_b --seeds 42 --smoke

# Ba seed chính thức, chỉ đánh giá validation
.\.venv\Scripts\python.exe -m har_project train --models cnn_b --seeds 42 123 2026
```

Các lượt đã tồn tại sẽ bị từ chối ghi đè để bảo toàn bằng chứng. Khi chỉ xem lại đồ án, dùng chế độ `inspect` trong notebook. Không chạy lại `prepare` đồng thời với tiến trình huấn luyện. Nếu cần chạy lại một cấu hình vì lý do nghiên cứu, ghi lý do trong protocol và chọn thư mục kết quả riêng bằng `--output`.

Trên máy khác, dùng Python 3.12 để tạo môi trường rồi cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` ghi các thư viện chính đã kiểm tra; `requirements-lock.txt` lưu đầy đủ các phiên bản môi trường đã cài. Không chép nguyên `.venv` sang máy khác. Để mở notebook trong VS Code, chọn interpreter `.venv/Scripts/python.exe`; thư mục dữ liệu cần giữ đúng như phần dưới.

## Dữ liệu và kết quả

Đầu vào CNN gồm chín tệp tín hiệu trong mỗi thư mục `train/Inertial Signals` và `test/Inertial Signals`; không dùng `X_train.txt` 561 đặc trưng làm đầu vào `(128,9)`. Validation dùng người 5, 11, 19, 26; fit-train/validation/test lần lượt có 5.982/1.370/2.947 cửa sổ.

- `har_project/data.py`: đọc, kiểm tra, chia tập, chuẩn hóa và tạo hình dữ liệu.
- `har_project/experiment.py`: CNN-A/B/C, baseline, huấn luyện và chỉ số validation.
- `results/data_summary.json`, `split.json`, `normalization.npz`: nguồn dữ liệu, SHA-256, index và thống kê chuẩn hóa.
- `results/prepared/dataset.npz`: cache mảng đã chuẩn hóa, được tạo bằng lệnh `prepare`, không theo dõi trong Git.
- `results/runs/cnn_b_seed*/`: cấu hình, môi trường, history, checkpoint, dự đoán và chỉ số validation của lượt chính thức.
- `results/smoke/`: lượt chạy thử; không trộn vào bảng kết quả chính thức.
- `results/figures/`: phân bố lớp và ví dụ tín hiệu thật.

Checkpoint `model.keras` được giữ trên máy và bỏ qua trong Git. Khi chuyển đồ án sang máy khác để đọc lại mô hình, cần chuyển riêng checkpoint tương ứng hoặc chạy huấn luyện lại; giữ cấu hình và lịch sử để đối chiếu.

Đánh giá test là mốc tiếp theo sau quyết định E2 và khóa protocol. Chỉ số validation hiện có không được ghi thành Accuracy test trong báo cáo.

## Tài liệu hiện có

- `De_cuong_NCKH_HAR_CNN_Ronao_Cho.docx`: đề cương sáu chương.
- `Paper/`: bài báo chính và tài liệu liên quan.
- `Dataset/UCI HAR Dataset/`: tín hiệu, đặc trưng, nhãn và thông tin người tham gia.

Các tài liệu Word/PDF gốc và dataset được giữ nguyên. Báo cáo hoàn chỉnh và slide sẽ được phát triển từ protocol, bản thảo và kết quả đã kiểm chứng.

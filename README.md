# SGU26_PPNCKH_Group

Đồ án môn Phương pháp nghiên cứu khoa học trong Khoa học dữ liệu: phân tích Ronao & Cho (2016) và tái hiện một phần CNN nhận dạng hoạt động con người trên UCI HAR.

## Bắt đầu

Để học, chạy và trình bày phần huấn luyện, mở [notebook huấn luyện từng bước](notebooks/02_har_training.ipynb). Mã đọc dữ liệu, định nghĩa CNN, compile, callbacks, `model.fit()`, lưu checkpoint và vẽ learning curves đều nằm trực tiếp trong các ô notebook.

[Notebook kiểm tra bộ thực nghiệm gốc](notebooks/01_har_experiment.ipynb) dùng để xác minh cache và xem các kết quả đã khóa. [Protocol thực nghiệm](docs/research/protocol.md) mô tả các quyết định nghiên cứu.

[Kế hoạch một tuần](docs/superpowers/plans/2026-11-09-har-research-project.md) và [nhật ký tiến độ](docs/research/progress.md) theo dõi các mốc. [Bảng dẫn chứng](docs/research/literature-matrix.md), [bảng đối chiếu](docs/research/original-vs-reproduction.md), [bản thảo Chương 1–3](report/ban_thao_chuong_1_3.md) và [Chương 4–5 với kết quả thật](report/chuong_4_5_ket_qua.md) phục vụ viết báo cáo.

Thực nghiệm E1/E2 đã hoàn tất: mỗi CNN-A/B/C có ba seed. CNN-B được chọn bằng validation trước test, đạt Accuracy test trung bình **91,53% ± 0,45 điểm phần trăm**, Macro-F1 **0,9161 ± 0,0046**. Baseline lớp phổ biến đạt Accuracy test 18,22%. Số liệu chi tiết nằm trong `results/metrics_summary.csv`; dấu ± là độ lệch chuẩn mẫu qua seed, không phải khoảng tin cậy giữa các người.

## Huấn luyện bằng notebook

1. Mở `notebooks/02_har_training.ipynb` trong VS Code/Jupyter và chọn kernel `.venv/Scripts/python.exe`.
2. Chạy các ô từ trên xuống. Mặc định `RUN_MODE="inspect"` hiển thị dữ liệu, kiến trúc, bảng và learning curves từ chín lượt gốc.
3. Muốn chạy thử, đổi `RUN_MODE="smoke"` và đặt `RUN_NAME` chưa dùng, rồi **Save → Restart Kernel → Run All**. Notebook huấn luyện CNN-B seed 42 trong một epoch và lưu checkpoint/dự đoán validation.
4. Muốn huấn luyện đầy đủ, đặt `RUN_MODE="train"`. Giữ `MODELS=["cnn_a", "cnn_b", "cnn_c"]` để chạy E2 hoặc chọn `["cnn_b"]` cho mô hình cơ sở; `SEEDS=[42,123,2026]` giữ ba lượt độc lập.

Kết quả mới nằm trong `notebook_runs/RUN_NAME/`, gồm cấu hình, bản notebook trên đĩa kèm SHA-256, thống kê chuẩn hóa, split, lịch sử, checkpoint, dự đoán và chỉ số validation. Notebook từ chối ghi đè một `RUN_NAME` đã tồn tại và không ghi vào `results/` gốc. Mỗi lần chạy mới sau khi đã xem test là tái chạy/khám phá; các số liệu đó không thay thế đợt đánh giá gốc đã khóa. Phần huấn luyện không nạp tín hiệu hoặc nhãn test.

Notebook huấn luyện không phụ thuộc hàm Python trong `har_project`. Nếu chạy trên môi trường khác, cài các thư viện trong `requirements.txt` và chỉnh `DATA_DIR` trong ô cấu hình tới UCI HAR.

## Chạy bổ sung bằng terminal trên Windows

Môi trường `.venv` trong workspace dùng Python 3.12 và đã có các thư viện cần thiết. Các lệnh dưới đây chạy từ thư mục gốc bằng PowerShell:

```powershell
# Kiểm tra các hành vi quan trọng của dữ liệu và mô hình
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Các bước chuẩn bị/huấn luyện sau áp dụng cho output mới, chưa khóa
.\.venv\Scripts\python.exe -m har_project prepare

# Chạy thử một epoch, kết quả nằm riêng trong results/smoke
.\.venv\Scripts\python.exe -m har_project train --models cnn_b --seeds 42 --smoke

# Ba seed chính thức, chỉ đánh giá validation
.\.venv\Scripts\python.exe -m har_project train --models cnn_b --seeds 42 123 2026

# E2: bổ sung hai cấu hình, giữ nguyên ba seed
.\.venv\Scripts\python.exe -m har_project train --models cnn_a cnn_c --seeds 42 123 2026

# Khóa quyết định từ validation, rồi tính test một đợt
.\.venv\Scripts\python.exe -m har_project freeze
.\.venv\Scripts\python.exe -m har_project evaluate

# Lệnh dùng để xem lại bộ kết quả hiện có; không chạy lại suy luận
.\.venv\Scripts\python.exe -m har_project verify
```

**Output `results/` hiện đã khóa và đã đánh giá test:** dùng `verify` hoặc chế độ `inspect` trong notebook để xem lại. Các lệnh prepare/train/freeze/evaluate sẽ từ chối ghi đè hoặc thay đổi đợt đã khóa. Không chạy lại `prepare` đồng thời với tiến trình huấn luyện. Nếu cần thực nghiệm mới, ghi lý do trong protocol khám phá và tạo output riêng; ví dụ `prepare --output results_reproduction`, rồi `train --prepared results_reproduction/prepared/dataset.npz --output results_reproduction --models cnn_a cnn_b cnn_c`. Không sửa mã/protocol đã được lock của đợt chính ghi nhận.

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
- `results/runs/cnn_*_seed*/`: cấu hình, môi trường, history, checkpoint, dự đoán và chỉ số validation/test của chín lượt chính thức.
- `results/smoke/`: lượt chạy thử; không trộn vào bảng kết quả chính thức.
- `results/figures/`: phân bố lớp và ví dụ tín hiệu thật.
- `results/protocol_lock.json`: quyết định chọn CNN-B từ validation và dấu kiểm tra của dữ liệu/mã/checkpoint trước test.
- `results/test_evaluation_status.json`: trạng thái hoàn tất và dấu kiểm tra của bộ kết quả cuối.
- `results/metrics_per_run.csv`, `metrics_summary.csv`: kết quả test từng seed và tổng hợp; các tệp tiền tố `validation_` là kết quả validation.
- `results/test_class_metrics.csv`, `test_subject_metrics.csv`, `test_error_pairs.csv`: chỉ số từng lớp/người và các cặp nhầm lẫn.
- `report/figures/`: năm hình phục vụ báo cáo, tạo từ kết quả đã xác minh bằng `python scripts/plot_report_figures.py`.

Checkpoint `model.keras` và cache chuẩn bị được giữ trên máy, bỏ qua trong Git. Để xác minh nguyên vẹn đợt đã khóa trên máy khác, cần chuyển riêng đủ chín checkpoint, cache `results/prepared/dataset.npz` và toàn bộ tệp được lock tham chiếu. Huấn luyện lại tạo một đợt mới; không thay checkpoint cũ bằng trọng số mới rồi coi đó là cùng bằng chứng.

Không dùng kết quả test đã xem để lựa chọn lại siêu tham số của cùng đợt. Các cải tiến nảy sinh từ phân tích lỗi được ghi là nghiên cứu khám phá và cần quy trình đánh giá mới.

## Tài liệu hiện có

- `De_cuong_NCKH_HAR_CNN_Ronao_Cho.docx`: đề cương sáu chương.
- `Paper/`: bài báo chính và tài liệu liên quan.
- `Dataset/UCI HAR Dataset/`: tín hiệu, đặc trưng, nhãn và thông tin người tham gia.

Các tài liệu Word/PDF gốc và dataset được giữ nguyên. Báo cáo hoàn chỉnh và slide sẽ được phát triển từ protocol, bản thảo và kết quả đã kiểm chứng.

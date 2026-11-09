# Nhật ký triển khai đồ án HAR

## Đợt khởi động ngày 09/11/2026

Đã thực hiện phần nghiên cứu và pipeline của M1–M4. E1 đã có ba lượt CNN-B và kết quả validation; **chưa hoàn tất đánh giá test**, chưa khóa quyết định E2. Báo cáo hoàn chỉnh và slide thuộc các mốc sau.

| Mốc | Trạng thái | Bằng chứng |
|---|---|---|
| M1 | Đã chốt protocol khoa học; thông tin hành chính còn chờ người dùng/giảng viên | `protocol.md`; chưa tự gán tên thành viên, hạn nộp hoặc rubric |
| M2 | Đã có bảng dẫn chứng, đối chiếu và bản thảo đầu | `literature-matrix.md`, `original-vs-reproduction.md`, `report/ban_thao_chuong_1_3.md` |
| M3 | Đã đọc, kiểm tra, chia người, chuẩn hóa và tạo hình dữ liệu | Notebook, `results/data_summary.json`, `split.json`, `normalization.npz`, `results/figures/` |
| M4 | Đã chạy smoke và ba seed chính thức, lưu và nạp lại checkpoint | `results/smoke/cnn_b_seed42/`, `results/runs/cnn_b_seed42/`, `cnn_b_seed123/`, `cnn_b_seed2026/` |
| M5 | Chưa thực hiện; đã có thời gian chạy thực tế để quyết định E2 | Xem bảng dưới; chưa tạo khóa đánh giá test |
| M6–M8 | Chưa thực hiện | Đánh giá test, báo cáo cuối và thuyết trình là công việc tiếp theo |

## Kết quả validation hiện có

Tập validation có 1.370 cửa sổ thuộc bốn người, cố định cho mọi seed. Mô hình CNN-B có 36.294 tham số. Early stopping theo validation loss, không theo Accuracy; các chỉ số dưới đây tính từ checkpoint có validation loss nhỏ nhất của từng lượt.

| Seed | Epoch đã chạy | Epoch checkpoint tốt nhất | Accuracy validation | Macro-F1 validation | Thời gian huấn luyện CPU |
|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 2 | 94,6715% | 0,950433 | 22,56 giây |
| 123 | 20 | 12 | 95,7664% | 0,959491 | 47,48 giây |
| 2026 | 16 | 8 | 94,6715% | 0,950584 | 35,37 giây |

Trung bình Accuracy validation **95,0365%**, độ lệch chuẩn mẫu **0,6321 điểm phần trăm**. Trung bình Macro-F1 **0,953502**, độ lệch chuẩn mẫu **0,005187**. Baseline lớp nằm chọn từ fit-train đạt Accuracy validation **19,5620%**, Macro-F1 **0,054538**.

Nguồn máy đọc: `results/validation_metrics_per_run.csv` và `results/validation_metrics_summary.csv`. Các chỉ số từng lượt đã được tính lại từ `validation_predictions.csv`; tổng mỗi confusion matrix là 1.370. Không đưa smoke vào trung bình. Kết quả này chưa được dùng để kết luận H1 trên test hoặc để so sánh trực tiếp với 94,79%/95,75% của bài gốc.

Learning curves cho thấy loss train tiếp tục giảm trong khi loss validation tăng về sau, rõ nhất ở seed 123 và 2026. Early stopping và checkpoint giữ lại trạng thái validation loss tốt hơn trước đó. Không lựa chọn lại learning rate, dropout hoặc seed dựa trên các kết quả test vì chưa tính điểm test.

## Kiểm tra và môi trường

- Môi trường riêng trong `.venv`, Python 3.12, TensorFlow 2.21.0; TensorFlow nhận CPU. Đã kiểm tra không có xung đột dependency bằng `pip check`.
- `requirements.txt` ghi các thư viện chính; `requirements-lock.txt` lưu đầy đủ phiên bản đã cài. Các tệp `environment.json` ghi cấu hình từng lượt.
- Có kiểm tra hành vi ghép kênh/nhãn, chia theo người, chuẩn hóa train-only, kênh hằng, baseline, Macro-F1 đủ sáu lớp, output mô hình và từ chối ghi đè run.
- Đã rà soát mã và sửa notebook: `inspect` chỉ đọc/xác minh cache, không tạo lại bằng chứng; Unicode tiếng Việt được lưu đúng UTF-8.
- Notebook đã chạy lại từ kernel sạch ở chế độ inspect; xác nhận bốn tệp chuẩn bị giữ nguyên thời điểm sửa đổi. Có regression test cho hành vi này và việc phát hiện dữ liệu nguồn thay đổi.
- Các SHA-256 của cache chuẩn bị khớp với cấu hình của cả ba lượt chính thức. Tệp dữ liệu nguồn và bài báo gốc được giữ nguyên.
- Sandbox chặn tạo nhánh Git; các đầu ra được viết trực tiếp trong workspace. Chưa tạo commit.

## Công việc kế tiếp

1. Dựa trên thời gian chạy CNN-B khoảng 23–47 giây/lượt trên môi trường hiện tại, E2 có thể thực hiện mà chưa cần chuyển GPU. Đây là ước lượng cho quyết định nguồn lực, không bảo đảm CNN-A/C có cùng thời gian.
2. Chạy CNN-A/C theo cùng ba seed nếu chọn E2, giữ nguyên protocol huấn luyện; sau đó khóa danh sách cấu hình, seed và checkpoint.
3. Thực hiện M6: đánh giá test một đợt theo thiết kế đã khóa, lưu dự đoán, tính chỉ số và tạo confusion matrix cho báo cáo.
4. Phát triển bản thảo Chương 1–3 thành báo cáo sáu chương, bổ sung kết quả thật vào Chương 4–5 và chuẩn bị slide.

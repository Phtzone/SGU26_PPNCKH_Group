# Nhật ký triển khai đồ án HAR

## Trình bày huấn luyện bằng notebook ngày 09/11/2026

Đã hợp nhất vào [notebook nghiên cứu duy nhất](../../notebooks/HAR_Research.ipynb), gồm nội dung nghiên cứu, mã NumPy đọc dữ liệu/chia theo người/chuẩn hóa, định nghĩa CNN-A/B/C, compile, callbacks, `model.fit()`, kết quả gốc và khảo sát cảm biến. Notebook này không gọi CLI hay hàm huấn luyện của `har_project`.

Mặc định `RUN_MODE="inspect"` đọc lại lịch sử và bảng của các lượt gốc. Đổi sang `smoke` để kiểm tra một epoch CNN-B seed 42, hoặc `train` để chạy các mô hình/seed đã cấu hình. Các đợt mới chỉ ghi vào `notebook_runs/RUN_NAME/`; từ chối ghi đè, thoát đường dẫn hoặc ghi vào `results/` gốc. Việc huấn luyện notebook không nạp tín hiệu/nhãn test, chỉ đánh giá validation.

Đã thực thi Run All từ kernel sạch cho chế độ inspect và một lượt smoke riêng. Smoke lưu được checkpoint, history và dự đoán; nạp lại checkpoint khớp mô hình trong bộ nhớ, Accuracy validation 0,941606 và Macro-F1 0,944524 sau một epoch. Các số này là kiểm tra kỹ thuật, không bổ sung vào kết quả chính thức. Có kiểm tra hành vi cho ba kiến trúc và giới hạn đường dẫn/ghi đè; bộ kết quả gốc tiếp tục được xác minh nguyên vẹn.

## Trạng thái sau đợt tiếp tục ngày 09/11/2026

**Đã hoàn thành M5–M6 và E1/E2:** chín lượt CNN-A/B/C theo ba seed đã huấn luyện, khóa quyết định trước test và đánh giá trên cùng 2.947 cửa sổ test. M7 đang được phát triển từ các bản thảo, chưa có báo cáo Word/PDF cuối; M8 chưa hoàn thành.

| Cấu hình | Accuracy test trung bình ± SD | Macro-F1 test trung bình ± SD | Số tham số |
|---|---:|---:|---:|
| CNN-A | 90,6911% ± 0,3079 điểm % | 0,908219 ± 0,003023 | 7.494 |
| CNN-B | 91,5281% ± 0,4519 điểm % | 0,916081 ± 0,004601 | 36.294 |
| CNN-C | 91,2566% ± 0,0784 điểm % | 0,912694 ± 0,000926 | 85.574 |

CNN-B được chọn trước khi xem test, bằng Macro-F1 validation trung bình 0,953502. Baseline lớp nằm xác định từ fit-train đạt Accuracy test 18,2219%, Macro-F1 0,051378. Cặp ngồi–đứng có số lỗi hai chiều cao nhất ở cả ba seed CNN-B: 197, 190 và 189. Accuracy theo người cho thấy mức thấp hơn ở ID 9/10; đây là mô tả kết quả, chưa xác định nguyên nhân cá nhân.

Bằng chứng mới:

- `results/protocol_lock.json`: ghi quyết định theo validation và danh sách đủ chín lượt; trước khi khóa đã nạp lại mọi checkpoint và đối chiếu xác suất validation, kiến trúc/số tham số.
- `results/test_evaluation_status.json`: đợt test hoàn tất, có dấu kiểm tra tệp kết quả. Các lượt chính chỉ tính test một đợt; việc xem lại dùng CSV và lệnh `verify`.
- `results/metrics_per_run.csv`, `metrics_summary.csv`, `test_class_metrics.csv`, `test_subject_metrics.csv`, `test_error_pairs.csv`: số liệu dùng viết kết quả.
- [Chương 4–5](../../report/chuong_4_5_ket_qua.md): thiết kế thực tế, kết quả từng seed, phân tích lớp/người, giả thuyết và đối chiếu bài gốc.
- `report/figures/`: năm hình báo cáo, có provenance; hình được xuất từ kết quả đã lưu, không chạy lại suy luận test.
- Notebook đã bổ sung chế độ đọc kết quả cuối; prepare/train trong output đã khóa bị chặn qua CLI và notebook.

Rà soát trước đánh giá test đã bổ sung hai kiểm soát: chứng minh checkpoint hiện tại khớp với bằng chứng validation, và từ chối lock thiếu danh sách tệp/quyết định bắt buộc trước khi đọc test. Không đổi kiến trúc, learning rate, dropout, split hoặc seed sau khi xem test.

Việc tiếp theo là ghép và hoàn thiện báo cáo sáu chương, chuẩn hóa trích dẫn/bố cục, tạo bản Word/PDF và slide. Các yêu cầu hành chính về rubric, hạn nộp chính xác, thành viên và thời lượng trình bày vẫn chưa được người dùng cung cấp.

Phần dưới giữ lại ảnh chụp tiến độ của đợt khởi động để truy nguyên; các dòng “chưa test” chỉ phản ánh thời điểm đó.

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

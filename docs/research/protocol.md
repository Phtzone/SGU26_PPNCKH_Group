# Protocol thực nghiệm CNN trên UCI HAR

Phiên bản 1.0, lập ngày 09/11/2026 trước các lượt huấn luyện chính thức. Thiết kế này thuộc nhóm thực hiện đồ án, được triển khai từ kế hoạch đã chốt trong `docs/superpowers/plans/2026-11-09-har-research-project.md`.

## Mục tiêu và phạm vi

Phân tích Ronao & Cho (2016), xây dựng CNN 1D giản lược trên UCI HAR và đánh giá khả năng nhận dạng sáu hoạt động ở những người không xuất hiện trong tập huấn luyện. Đầu ra bắt buộc là E1: CNN-B, baseline lớp phổ biến nhất và phân tích lỗi từng lớp. E2 so sánh CNN-A/B/C được thực hiện khi nguồn lực cho phép. SVM, FFT, thử 6 kênh và CNN–LSTM nằm ngoài phần bắt buộc trong tuần này.

Đây là tái hiện một phần nguyên lý học đặc trưng từ chuỗi cảm biến. CNN của nhóm có chín kênh, khác sáu kênh mô tả trong bài gốc; kiến trúc và quy trình tối ưu hóa cũng khác.

## Câu hỏi và giả thuyết

| Mã | Phát biểu | Bằng chứng / điều kiện kết luận |
|---|---|---|
| RQ1 | Vì sao CNN phù hợp với tín hiệu cảm biến? | Phân tích cơ chế cục bộ theo thời gian, chia sẻ trọng số và học phân cấp trong bài gốc |
| RQ2 / H1 | CNN-B vượt dự đoán lớp phổ biến nhất trên dữ liệu người mới | Đối chiếu Accuracy và Macro-F1 trên cùng test; báo cáo từng seed và trung bình |
| RQ3 / H2 | Ngồi–đứng là một cặp nhầm lẫn nổi bật | Confusion matrix, tổng lỗi hai chiều theo cặp và Recall từng lớp; không mặc định giả thuyết đúng |
| RQ4 / H3 | Kết quả thay đổi giữa ba cấu hình độ sâu | E2 với cùng split, seed và các tham số huấn luyện; không coi là cô lập hoàn toàn tác động độ sâu |
| RQ5 | Có thể đối chiếu kết quả nhóm với bài gốc tới mức nào? | Bảng đối chiếu các điều kiện thí nghiệm và giới hạn |

Không đặt ngưỡng đạt 94,79% hay 95,75% làm tiêu chí thành công. Nếu E2 không thực hiện, H3 được ghi là chưa kiểm chứng. Không kết luận về H4/SVM khi chưa có E3.

## Dữ liệu và chia tập cố định

- Dữ liệu nguồn: `Dataset/UCI HAR Dataset/`, bản đã có trong workspace.
- Cửa sổ dài 128 mẫu, tần số 50 Hz, tương đương 2,56 giây; đã được dataset xử lý và phân đoạn với mức chồng lấn 50%.
- Thứ tự kênh: body_acc X/Y/Z → body_gyro X/Y/Z → total_acc X/Y/Z.
- Đầu vào `float32`, dạng `(N,128,9)`. Chuyển nhãn nguồn 1–6 thành 0–5 và giữ tên lớp khi xuất báo cáo.
- Validation gồm người **5, 11, 19, 26**, chọn trước khi huấn luyện. Những người còn lại trong train chính thức tạo thành fit-train.
- Fit-train: 5.982 cửa sổ, 17 người. Validation: 1.370 cửa sổ, 4 người. Test chính thức: 2.947 cửa sổ, 9 người.
- Giữ nguyên thứ tự mẫu; không chia ngẫu nhiên các cửa sổ chồng lấn giữa fit-train và validation.
- Không huấn luyện lại trên toàn bộ 7.352 mẫu ở giai đoạn cuối trong protocol 1.0.

## Chuẩn hóa và kiểm tra

Kiểm tra số dòng, độ dài cửa sổ, nhãn, subject và giá trị NaN/Inf trước huấn luyện. Bắt buộc không giao người giữa ba tập và có đủ sáu lớp ở mỗi tập.

Với mỗi kênh, fit mean và standard deviation trên hai trục mẫu/thời gian của fit-train. Tính thống kê bằng `float64`, thay standard deviation dưới `1e-8` bằng `1e-8`, áp dụng cùng tham số cho cả ba tập rồi chuyển đầu vào thành `float32`. Lưu `normalization.npz`, index split và SHA-256 các tệp đầu vào để truy nguyên.

## Mô hình và huấn luyện

CNN-B: Conv1D 64/kernel 5 → MaxPool 2 → Conv1D 128/kernel 3 → GlobalAveragePooling1D → Dense 64 ReLU → Dropout 0,3 → Dense 6 Softmax. Mọi Conv dùng ReLU và padding `same`.

CNN-A bỏ Conv 128 của CNN-B. CNN-C thêm một Conv 128/kernel 3 sau Conv thứ hai. Phần cuối và pooling đầu tiên được giữ nguyên.

| Tham số | Giá trị |
|---|---|
| Optimizer / learning rate | Adam / 0,001 |
| Batch size | 64 |
| Giới hạn epoch | 50 |
| Loss | Sparse categorical cross-entropy |
| Early stopping | `val_loss`, patience 8, khôi phục trọng số tốt nhất |
| Checkpoint | Mô hình tại `val_loss` thấp nhất |
| Seed chính thức | 42, 123, 2026 |
| Xáo trộn | Chỉ fit-train, seed cố định cho từng lượt |

Đặt seed cho Python, NumPy và TensorFlow; bật toán tử quyết định được khi được hỗ trợ. Ghi môi trường và giới hạn tính tái lập. Mỗi lượt khởi tạo lại mô hình/optimizer, không tiếp tục từ trọng số của seed trước.

Chạy thử một epoch có mã `smoke` và thư mục riêng. Không trộn chỉ số smoke vào kết quả chính thức. Không ghi đè các lượt chạy đã lưu.

## Baseline và đánh giá

Baseline chọn lớp phổ biến nhất bằng nhãn fit-train, là lớp nằm theo split đã định. Trước khi khóa mô hình, chỉ đánh giá baseline và CNN trên validation. Không tính điểm test trong bước chuẩn bị hoặc huấn luyện.

Sau khi khóa danh sách cấu hình, seed và checkpoint, mới đánh giá test. Báo cáo Accuracy, Macro-F1, Precision/Recall/F1/support từng lớp, confusion matrix và thời gian huấn luyện. Các chỉ số dùng đủ sáu nhãn và `zero_division=0`.

Với ba seed, giữ bảng từng lượt rồi tính mean và standard deviation mẫu (`ddof=1`). Không chọn seed tốt nhất để đại diện toàn bộ phương pháp. Hình confusion matrix chính định trước dùng CNN-B seed 42. Biến động qua seed không thay thế đánh giá trên nhiều split/người và không tự tạo thành kiểm định ý nghĩa thống kê.

Nếu thực hiện E2, chọn cấu hình đại diện theo trung bình Macro-F1 validation qua cùng ba seed; hòa tới bốn chữ số thập phân thì ưu tiên ít tham số hơn. Số lớp, số tham số và receptive field cùng thay đổi nên kết luận giới hạn ở ba cấu hình đã thử.

## Sản phẩm và các yêu cầu chưa có thông tin

Lưu notebook, mã hỗ trợ, cấu hình mỗi lượt, môi trường, history, checkpoint, validation metrics, chỉ số/dự đoán test sau khóa, hình và bảng tổng hợp. Báo cáo và slide lấy số liệu từ các tệp này.

Người dùng đã xác nhận thời gian còn khoảng một tuần. Workspace chưa có hạn nộp chính xác, rubric, số thành viên, tên người phụ trách, thời lượng thuyết trình và quy định định dạng nộp. Các thông tin hành chính này chưa được suy đoán; chúng không cản trở xử lý dữ liệu và chạy E1. Tạm tổ chức công việc theo ba vai trò: học thuật, thực nghiệm, biên tập/thuyết trình, chưa gán tên cá nhân.

## Nhật ký thay đổi

| Phiên bản | Thay đổi | Thời điểm so với đánh giá test |
|---|---|---|
| 1.0 | Cụ thể hóa protocol đã chốt; triển khai đọc dữ liệu, chia người và chuẩn hóa; chưa đổi mục tiêu hoặc siêu tham số | Trước huấn luyện chính thức và trước đánh giá test |


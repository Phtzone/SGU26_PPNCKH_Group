# Câu hỏi thực tế về nhận dạng hoạt động

Thiết kế mở rộng ngày 10/11/2026, sau khi đã xem test của protocol 1.0. Đây là nghiên cứu khám phá bổ sung; không thay thế kết quả gốc và không có test độc lập mới.

## Thiết kế đã chốt

1. **Người mới có được nhận dạng tốt không?** Đọc ba bảng dự đoán test CNN-B đã lưu và được xác minh SHA-256. Báo cáo Accuracy/Macro-F1 theo người, trung bình qua seed và độ lệch chuẩn mẫu. Không chạy lại suy luận test, không dùng người test để điều chỉnh mô hình. Không suy luận nguyên nhân cá nhân từ ID.
2. **Ngồi và đứng bị nhầm như thế nào?** Tính lỗi hai chiều, Recall và tỷ trọng lỗi từ dự đoán test cũ. Minh họa tín hiệu bằng cửa sổ đầu tiên của mỗi lớp ở người validation ID 5, cùng quy tắc chọn trước khi xem hình. Mô tả phân phối RMS gia tốc cơ thể và trung bình gia tốc tổng từ official train. Các hình gợi ý cơ chế, không chứng minh nguyên nhân lỗi hoặc đại diện mọi người.
3. **Chỉ gia tốc kế có đủ không, con quay và các kênh bổ sung mang lại gì?** Huấn luyện lại CNN-B với năm đầu vào, mỗi đầu vào ba seed 42/123/2026:

| Mã | Kênh |
|---|---|
| total_acc | total_acc_x/y/z |
| gyro | body_gyro_x/y/z |
| total_acc_gyro | total_acc_x/y/z + body_gyro_x/y/z |
| body_acc_gyro | body_acc_x/y/z + body_gyro_x/y/z |
| all_9 | body_acc_x/y/z + body_gyro_x/y/z + total_acc_x/y/z |

Đây là thí nghiệm về **biểu diễn đầu vào**, không phải phép đo năng lượng của phần cứng. Gia tốc cơ thể và gia tốc tổng xuất phát từ cùng cảm biến gia tốc, không phải hai cảm biến vật lý riêng. Không gọi ba kênh gia tốc cơ thể là tín hiệu gia tốc thô.

## Kiểm soát thực nghiệm

- Chỉ tải tín hiệu official train cho thực nghiệm mới. Fit-train 17 người, validation 4 người ID 5/11/19/26; đủ sáu lớp và không trùng người.
- Chuẩn hóa từng kênh bằng mean/std fit-train; cùng phân chia, seed, Adam 0,001, batch 64, max 50 epoch, dropout 0,3, patience 8 cho mọi cấu hình. Checkpoint theo val_loss thấp nhất.
- CNN-B giữ phần còn lại như protocol 1.0. Số tham số lớp đầu thay đổi theo số kênh, phải báo cáo. Dùng cùng seed không bảo đảm cùng trọng số ban đầu giữa các kích thước đầu vào.
- Validation dùng cho early stopping và phân tích; các điểm mới là điểm validation phát triển, không phải ước lượng test độc lập. Không kiểm định ý nghĩa từ ba seed, không chọn seed tốt nhất.
- Báo cáo cả 15 lượt, mean/SD ddof=1, Recall từng lớp, số lỗi ngồi–đứng và chênh lệch ghép theo seed. Không khẳng định cải thiện nếu bằng chứng không nhất quán.
- Chế độ smoke chỉ một seed/một epoch, lưu riêng, không trộn vào nghiên cứu đầy đủ.
- Notebook hiển thị mã huấn luyện trực tiếp; mặc định inspect chỉ đọc. Đầu ra riêng `practical_results/`, từ chối ghi đè; giữ nguyên `results/` và mã nguồn được lock.
- Lưu cấu hình trước huấn luyện, snapshot notebook, hash đầu vào, normalization, split, lịch sử, checkpoint, dự đoán, metrics, hình và manifest kết quả. Inspect xác minh manifest và tính lại metrics từ CSV.

## Sản phẩm

- [Notebook thực nghiệm](../../notebooks/03_practical_questions.ipynb).
- `practical_results/sensor_study_v1/`: 15 lượt và các bảng/hình trả lời.
- `practical_results/sensor_study_v1/answers.md`: câu trả lời có số liệu, cách nói khi thuyết trình và giới hạn.

Không mở rộng kết luận sang điện thoại trong túi quần, thiết bị khác, đo pin, triển khai thời gian thực hoặc nguyên nhân sinh lý cá nhân.

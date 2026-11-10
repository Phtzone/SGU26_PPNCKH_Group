# Câu trả lời thực tế từ thực nghiệm HAR

Nghiên cứu khám phá bổ sung sau khi đã xem test của protocol 1.0. Test cũ chỉ được đọc từ dự đoán lưu sẵn; thí nghiệm cảm biến mới chỉ đánh giá validation.

## 1. Người mới có được nhận dạng tốt không?

CNN-B đạt Accuracy test trung bình 91.53% qua ba seed trên cùng 9 người chưa xuất hiện trong huấn luyện. Accuracy trung bình theo từng người nằm trong khoảng 71.88%–99.91%.

**Cách giải thích:** mô hình có khả năng nhận dạng người mới trong bộ dữ liệu này, nhưng chất lượng không đồng đều. Điểm trung bình không bảo đảm mọi người đều đạt mức đó. Chưa có bằng chứng về điện thoại, vị trí đeo hoặc bối cảnh khác.

## 2. Ngồi và đứng bị nhầm như thế nào?

Số lỗi ngồi–đứng hai chiều qua ba seed là [197, 190, 189]; chiếm 74.3%–78.4% tổng lỗi của từng seed.

**Cách giải thích:** đây là khó khăn nổi bật quan sát được. Khi ít chuyển động, gia tốc cơ thể có thể cung cấp ít dấu hiệu phân biệt; gia tốc tổng có thể bổ sung thông tin tư thế. Đây là giả thuyết cơ chế, không phải nguyên nhân đã được chứng minh. Hình ví dụ lấy từ validation, không phải cửa sổ test bị nhầm.

## 3. Chỉ gia tốc kế có đủ không?

Các điểm sau dùng cùng một validation gồm bốn người, cũng được dùng cho early stopping. Chúng chưa phải bằng chứng test độc lập:

| Biểu diễn | Accuracy mean ± SD | Macro-F1 mean ± SD | Lỗi ngồi–đứng TB | Tham số |
|---|---|---|---|---|
| total_acc | 91.48% ± 2.02 điểm % | 0.9162 ± 0.0206 | 49.7 | 34,374 |
| gyro | 75.67% ± 1.14 điểm % | 0.7696 ± 0.0071 | 54.7 | 34,374 |
| total_acc_gyro | 92.58% ± 0.43 điểm % | 0.9277 ± 0.0048 | 51.3 | 35,334 |
| body_acc_gyro | 92.29% ± 0.64 điểm % | 0.9306 ± 0.0054 | 35.3 | 35,334 |
| all_9 | 95.04% ± 0.63 điểm % | 0.9535 ± 0.0052 | 50.3 | 36,294 |

**So sánh ghép cùng seed:** giá trị dương của ΔMacro-F1 có lợi cho cấu hình bổ sung; giá trị âm của Δlỗi ngồi–đứng nghĩa là ít nhầm hơn.

- total_acc → total_acc_gyro: ΔMacro-F1 trung bình +0.0115; khoảng qua seed [-0.0060, +0.0249], khác nhau theo seed hoặc có seed không đổi; Δlỗi ngồi–đứng trung bình +1.7.
- body_acc_gyro → all_9: ΔMacro-F1 trung bình +0.0229; khoảng qua seed [+0.0139, +0.0305], nhất quán tăng ở tất cả seed; Δlỗi ngồi–đứng trung bình +15.0.
- total_acc_gyro → all_9: ΔMacro-F1 trung bình +0.0258; khoảng qua seed [+0.0235, +0.0273], nhất quán tăng ở tất cả seed; Δlỗi ngồi–đứng trung bình -1.0.

**Giới hạn:** số kênh cũng thay đổi tham số lớp đầu; cùng seed không bảo đảm cùng khởi tạo khi kích thước khác nhau. Không đo pin, không suy ra hiệu quả trên điện thoại thật, không khẳng định cảm biến nào luôn tốt nhất. Tổng gia tốc và gia tốc cơ thể là hai biểu diễn từ cùng cảm biến gia tốc.

## Bằng chứng để kiểm tra

- `archived_subject_metrics.csv`, `archived_subject_summary.csv`, `archived_error_metrics.csv`: phân tích test cũ.
- `validation_runs.csv`, `validation_summary.csv`, `paired_sensor_deltas.csv`, `validation_subject_metrics.csv`: thí nghiệm mới.
- `runs/`: cấu hình, history, checkpoint và dự đoán từng lượt; `manifest.json`: SHA-256 toàn bộ hiện vật.
- `figures/`: sáu hình từ dữ liệu và dự đoán; `signal_features_train.csv`: đặc trưng mô tả official train.

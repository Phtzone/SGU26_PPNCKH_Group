# Đối chiếu công trình gốc và thực nghiệm của nhóm

Nguồn gốc: Ronao & Cho (2016), `Paper/Human_activity_recognition_with_smartpho.pdf`. Thiết kế nhóm: `protocol.md`. Bảng mô tả thiết kế trước khi có kết quả cuối.

| Thành phần | Công trình gốc | Thiết kế nhóm | Hệ quả khi so sánh |
|---|---|---|---|
| Dữ liệu chính | UCI HAR, 30 người, sáu hoạt động | Cùng bộ dữ liệu và sáu nhãn | Chung bài toán, chưa đủ để gọi là cùng điều kiện |
| Biểu diễn | 128 thời điểm, 6 kênh theo §4.1/Bảng 1 | 128 thời điểm, 9 kênh có sẵn | Có khác biệt thông tin đầu vào; không khẳng định tái hiện chính xác |
| Train / test | 7.352 / 2.947 từ 21/9 người | Giữ test; dành 1.370 mẫu/4 người làm validation, fit-train còn 5.982 | Ít dữ liệu cập nhật trọng số hơn; không huấn luyện lại trên cả train trong protocol 1.0 |
| Validation | Có dùng validation và lưu mô hình tốt; chưa rõ toàn bộ cách tách | Cố định người 5, 11, 19, 26, tách khỏi fit-train | Nhóm công khai split để tái lập; không tự gán cách chia cho tác giả |
| Chuẩn hóa | Có mô tả z-score; phạm vi fit thống kê chưa rõ trong phần được đọc | Theo từng kênh trên fit-train, áp dụng nguyên trạng cho holdout | Nêu điều đã biết và điều chưa xác minh; không suy ra bài gốc có rò rỉ |
| CNN không FFT tốt nhất | 96–192–192 feature maps, Dense 1000, kernel 9, pooling 3 | CNN-B 64–128, kernel 5/3, pooling 2, GAP, Dense 64 | Khác số lớp, tham số, receptive field và cơ chế tổng hợp |
| Tối ưu | SGD/momentum; nhiều cấu hình; tối đa 5.000 epoch | Adam 0,001; tối đa 50 epoch; patience 8 | So sánh kết quả như đánh giá triển khai giản lược |
| FFT | Cấu hình riêng, Accuracy 95,75% ở Bảng 5 | Không đưa FFT vào E1 | Không dùng 95,75% làm ngưỡng phải đạt |
| Chỉ số | Accuracy, precision/recall, confusion matrix | Bổ sung Macro-F1, từng seed, mean/std, thời gian | Bổ sung góc nhìn đánh giá, không thay đổi số gốc |
| Baseline | Các phương pháp trong Bảng 5, HCF+SVM 94,61% | Lớp phổ biến nhất trong fit-train | Baseline tối thiểu chưa đủ để kết luận vượt phương pháp truyền thống mạnh |
| Khảo sát | Độ sâu, feature maps, kernel, pooling, learning rate | E2 có điều kiện với ba cấu hình | Không nhận đã tái hiện toàn bộ khảo sát |
| Phạm vi bằng chứng | Có dataset bổ sung từ ba người ngoài thí nghiệm chính | Một dataset trong tuần này | Ghi đúng giới hạn của nhóm và của từng thí nghiệm tác giả |

Kết luận đối chiếu phải trả lời ba câu: khác biệt nào đã được kiểm soát, khác biệt nào có thể ảnh hưởng kết quả, và phát biểu nào dữ liệu hiện có chưa cho phép. Chênh lệch Accuracy riêng lẻ không chứng minh mô hình nào vượt trội trong điều kiện tương đương.


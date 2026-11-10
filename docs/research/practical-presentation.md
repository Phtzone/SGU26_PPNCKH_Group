# Giải thích các câu hỏi thực tế khi thuyết trình

Toàn bộ nội dung giải thích dưới đây đã được tích hợp vào mục 8 của [notebook nghiên cứu duy nhất](../../notebooks/HAR_Research.ipynb). Tệp này giữ bản hướng dẫn dạng văn bản; số liệu truy nguyên từ [câu trả lời được tạo từ dữ liệu](../../practical_results/sensor_study_v1/answers.md).

## Người mới sử dụng thì mô hình có còn đúng không?

**Cách nói:** “Chúng tôi tách người tham gia giữa huấn luyện và đánh giá. Trên chín người test chưa xuất hiện trong huấn luyện, CNN-B đạt Accuracy trung bình 91,53% qua ba seed. Tuy nhiên, Accuracy trung bình theo từng người nằm trong khoảng 71,88% đến 99,91%. Vì vậy, điểm trung bình cao chưa bảo đảm chất lượng đồng đều cho mọi người dùng.”

Dùng [biểu đồ theo người](../../practical_results/sensor_study_v1/figures/unseen_people.png). Các thanh sai số mô tả biến động qua seed trên cùng người, không phải khoảng tin cậy cho toàn bộ dân số.

**Nếu được hỏi vì sao người ID 10 thấp:** chưa đủ thông tin để xác định nguyên nhân. Không gán cho tuổi, sức khỏe, cách đeo hoặc cách đi khi dữ liệu chưa cung cấp bằng chứng tương ứng.

## Vì sao hay nhầm ngồi với đứng?

**Cách nói:** “Ngồi–đứng là khó khăn nổi bật của mô hình: lỗi hai chiều chiếm khoảng 74,3–78,4% tổng lỗi ở từng seed. Phân phối RMS gia tốc cơ thể của hai hoạt động chồng lấn nhiều. Đây là cơ sở đặt giả thuyết rằng tín hiệu chuyển động đơn thuần chưa đủ phân biệt tốt tư thế; chúng tôi kiểm tra thêm vai trò của các biểu diễn cảm biến.”

Dùng [confusion matrix](../../practical_results/sensor_study_v1/figures/sitting_standing_confusion.png) để chứng minh lỗi, [phân phối tín hiệu](../../practical_results/sensor_study_v1/figures/static_signal_features.png) để giải thích giả thuyết. Hai hình có vai trò khác nhau: hình phân phối chưa chứng minh nguyên nhân gây lỗi của CNN.

[Hình tín hiệu mẫu](../../practical_results/sensor_study_v1/figures/sitting_standing_signals.png) lấy cửa sổ đầu tiên của từng lớp ở validation ID 5 theo quy tắc cố định. Cửa sổ vẫn có thể chứa biến thiên hoặc quá trình ổn định tư thế; không tuyên bố mọi cửa sổ ngồi/đứng đều phẳng. Đây cũng không phải chính cửa sổ test bị dự đoán sai.

## Chỉ gia tốc kế có đủ không? Thêm cảm biến có luôn tốt hơn?

Đã chạy năm biểu diễn × ba seed, cùng CNN-B và cùng validation bốn người:

| Đầu vào | Macro-F1 trung bình ± SD | Accuracy trung bình |
|---|---|---|
| Gia tốc tổng, 3 kênh | 0,9162 ± 0,0206 | 91,48% |
| Con quay, 3 kênh | 0,7696 ± 0,0071 | 75,67% |
| Gia tốc tổng + con quay, 6 kênh | 0,9277 ± 0,0048 | 92,58% |
| Gia tốc cơ thể + con quay, 6 kênh | 0,9306 ± 0,0054 | 92,29% |
| Toàn bộ 9 kênh | 0,9535 ± 0,0052 | 95,04% |

**Cách nói:** “Chỉ dùng gia tốc tổng vẫn nhận dạng được khá tốt trên validation này. Kết hợp với con quay tăng Macro-F1 trung bình khoảng 0,0115, nhưng có một seed bị giảm, nên chúng tôi không kết luận thêm con quay luôn có lợi. Cấu hình 9 kênh đạt điểm trung bình cao nhất trong năm cấu hình được khảo sát.”

Dùng [biểu đồ cảm biến](../../practical_results/sensor_study_v1/figures/sensor_comparison.png) và [chênh lệch cùng seed](../../practical_results/sensor_study_v1/paired_sensor_deltas.csv). Không gọi việc tăng điểm trung bình là cải thiện có ý nghĩa thống kê; chưa thực hiện kiểm định phù hợp hoặc đánh giá nhiều split.

**Vì sao 6 kênh ít lỗi ngồi–đứng hơn mà Macro-F1 lại thấp hơn 9 kênh?** Cấu hình gia tốc cơ thể + con quay có trung bình 35,3 lỗi ngồi–đứng, thấp hơn 50,3 lỗi của 9 kênh. Tuy nhiên, nó nhầm ngồi→nằm trung bình 41,7 cửa sổ, so với 10,3 cửa sổ của 9 kênh. Recall lớp nằm tương ứng là 91,79% và 100%; Recall lớp ngồi là 79,57% và 92,61%. Nhìn riêng một cặp nhầm lẫn sẽ bỏ sót lỗi sang lớp khác.

Đây là phản ví dụ hữu ích cho giả thuyết ban đầu: bổ sung gia tốc tổng vào gia tốc cơ thể + con quay tăng Macro-F1 ở cả ba seed, **nhưng số lỗi ngồi–đứng hai chiều cũng tăng ở cả ba seed**. Không thể dùng kết quả này để tuyên bố đã giải quyết được nhầm ngồi–đứng.

“Đủ dùng” cho một sản phẩm còn phụ thuộc ngưỡng chấp nhận lỗi, người dùng mục tiêu, thiết bị và bối cảnh. Thí nghiệm hiện tại chưa kiểm chứng các yêu cầu đó. Số tham số cũng thay đổi nhẹ theo đầu vào: 34.374 / 35.334 / 36.294 cho 3 / 6 / 9 kênh.

## Những câu hỏi phản biện cần phân biệt

| Câu hỏi | Cách trả lời đúng phạm vi |
|---|---|
| Có dùng được trên mọi điện thoại hoặc bỏ điện thoại trong túi quần không? | Chưa được kiểm chứng. Dữ liệu nghiên cứu có bối cảnh thiết bị/vị trí đeo cụ thể. |
| Ít kênh có tiết kiệm pin hơn không? | Nghiên cứu chưa đo năng lượng trên thiết bị; không suy ra mức tiết kiệm pin từ số kênh. |
| Gia tốc cơ thể và gia tốc tổng là hai cảm biến khác nhau? | Đây là hai biểu diễn liên quan đến cùng cảm biến gia tốc; gia tốc cơ thể đã qua bước tách thành phần trọng lực của dataset. |
| Vì sao điểm validation khác test? | Hai tập có những người khác nhau; điểm phản ánh hai tập cụ thể. Chưa cô lập nguyên nhân của toàn bộ chênh lệch. |
| CNN đã được chứng minh tốt hơn SVM chưa? | Chưa. Nghiên cứu gốc của nhóm mới có baseline lớp phổ biến nhất, chưa có so sánh SVM cùng thiết kế. |
| Ít lỗi ngồi–đứng hơn có chắc mô hình tốt hơn không? | Không đủ kết luận: mô hình có thể nhầm những mẫu đó sang lớp khác. Phải xem cả Recall từng lớp, Macro-F1 và confusion matrix. |
| Ba seed có chứng minh kết quả đúng với mọi cách chia dữ liệu không? | Không. Chúng mô tả biến động khởi tạo trên cùng một split, chưa thay thế đánh giá trên nhiều nhóm người. |

Thí nghiệm cảm biến bổ sung được thiết kế sau khi đã xem test gốc và chỉ đánh giá validation. Khi trình bày, phải nói rõ ranh giới giữa phân tích test cũ và thực nghiệm phát triển mới.

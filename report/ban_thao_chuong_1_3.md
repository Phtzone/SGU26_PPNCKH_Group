# Bản thảo học thuật ban đầu của đồ án HAR

Bản làm việc ngày 09/11/2026 để phát triển thành báo cáo sáu chương. Nội dung dưới đây gồm bản nháp Chương 1 và cấu trúc có luận điểm cho Chương 2–3; chưa phải báo cáo cuối và chưa trình bày kết quả của nhóm.

## Chương 1 Giới thiệu và xác định vấn đề nghiên cứu

### 1.1 Bối cảnh và vấn đề

Nhận dạng hoạt động con người (Human Activity Recognition, HAR) là bài toán suy ra hoạt động từ các quan sát cảm biến. Trong phạm vi đồ án, mỗi quan sát là một cửa sổ tín hiệu gia tốc và vận tốc góc, còn nhãn là một trong sáu hoạt động: đi bộ, lên cầu thang, xuống cầu thang, ngồi, đứng và nằm. Dữ liệu có cấu trúc chuỗi thời gian nhiều biến; cách thực hiện cùng một hoạt động khác nhau giữa các cá nhân, làm cho đánh giá trên người chưa được huấn luyện trở thành yêu cầu quan trọng.

Các phương pháp dựa trên đặc trưng thủ công mô tả mỗi cửa sổ bằng các đại lượng thống kê hoặc tần số rồi phân loại. CNN tạo một hướng tiếp cận khác: học các bộ lọc từ tín hiệu và kết hợp đặc trưng qua nhiều lớp. Ronao và Cho (2016) khai thác cấu trúc cục bộ theo thời gian và tính phân cấp của hoạt động để xây dựng CNN cho HAR. Đây là cơ sở lựa chọn công trình cho việc phân tích phương pháp nghiên cứu trong đồ án [R1].

Vấn đề của đồ án được giới hạn ở việc hiểu cách công trình hình thành giải pháp, kiểm tra CNN giản lược có học được tín hiệu phân biệt sáu lớp hay không, và giải thích các sai số xuất hiện. Việc huấn luyện một mô hình khác kiến trúc gốc không đủ để xác nhận hay bác bỏ toàn bộ kết luận của bài báo.

### 1.2 Mục tiêu

Mục tiêu tổng quát là phân tích nghiên cứu Ronao và Cho, sau đó tái hiện một phần nguyên lý CNN trên UCI HAR. Các mục tiêu cụ thể gồm: làm rõ lý do lựa chọn CNN; mô tả thiết kế và bằng chứng của bài gốc; xây dựng pipeline tránh rò rỉ giữa fit-train, validation và test; đánh giá từng lớp và so với baseline; đối chiếu khác biệt giữa hai thiết lập.

### 1.3 Câu hỏi và giả thuyết

RQ1 phân tích sự phù hợp của CNN với chuỗi cảm biến. RQ2 xác định hiệu quả nhận dạng của CNN-B và kiểm tra H1 rằng mô hình vượt dự đoán lớp phổ biến nhất. RQ3 tìm các cặp hoạt động bị nhầm, trong đó H2 dự kiến ngồi–đứng là một cặp nổi bật. Các giả thuyết này do nhóm đặt ra và chỉ được kết luận bằng dữ liệu thực nghiệm.

RQ4 khảo sát ba cấu hình có độ sâu khác nhau nếu thời gian cho phép; H3 chỉ được đánh giá khi có đủ lượt chạy tương ứng. RQ5 xem xét mức độ có thể đối chiếu kết quả với bài gốc, dựa trên dữ liệu đầu vào, kiến trúc, huấn luyện và cách đánh giá. Bằng chứng và quy tắc quyết định được ghi trong `docs/research/protocol.md` trước khi chạy chính thức.

### 1.4 Đối tượng và phạm vi

Đối tượng là phương pháp phân loại hoạt động từ dữ liệu smartphone bằng CNN. Bộ dữ liệu sử dụng là UCI HAR đã có trong workspace. Nhóm dùng chín kênh chuỗi đã được xử lý và phân cửa sổ, dành bốn người trong train chính thức làm validation, và giữ nguyên test của dataset. Mô hình này là thiết kế giản lược của nhóm; bài gốc mô tả sáu kênh đầu vào [R1, §4.1].

Đồ án không thu thập dữ liệu mới hoặc triển khai ứng dụng điện thoại trong phạm vi một tuần. Khả năng sử dụng trên thiết bị và bối cảnh khác được xem là giới hạn cần thảo luận. Các kết quả thực nghiệm sẽ được báo cáo khi có tệp dự đoán, lịch sử huấn luyện và bảng chỉ số có thể kiểm tra lại.

## Chương 2 Tổng quan và cơ sở lý thuyết

### 2.1 Các khái niệm cần phát triển

Giải thích HAR dựa trên cảm biến, cửa sổ thời gian nhiều biến, gia tốc tổng/gia tốc cơ thể và vận tốc góc. Dùng README của dataset và nguồn tạo dữ liệu [R2] để mô tả nguồn gốc. Phân biệt chín kênh được cung cấp với 561 đặc trưng tính sẵn và với số trục cảm biến vật lý.

### 2.2 Lập luận về CNN

Trình bày convolution 1D dọc trục thời gian, chia sẻ trọng số, receptive field và sự kết hợp đặc trưng qua nhiều lớp. Sau đó giải thích ReLU, max-pooling, GAP, softmax và cross-entropy theo vai trò trong mô hình nhóm. Pooling và GAP có thể làm mất thông tin; không mô tả chúng như bảo đảm nhận dạng luôn tốt hơn.

### 2.3 Tổng quan nghiên cứu liên quan

Tổ chức các công trình theo vai trò: nguồn tạo UCI HAR [R2], CNN cho dữ liệu smartphone [R1], hướng kết hợp CNN–LSTM [R3], so sánh cảm biến phone/watch [R4], và các thách thức tổng quan [R5]. Các dataset khác nhau được dùng để mở rộng lập luận, không xếp hạng thuật toán bằng điểm số trực tiếp giữa các bài.

Khoảng trống ở phạm vi đồ án là kiểm tra một triển khai giản lược có minh bạch dữ liệu, split, seed và phân tích lỗi. Không phát biểu rằng nhóm đã tìm thấy một khoảng trống mới của toàn lĩnh vực nếu chưa có tổng quan đủ rộng.

## Chương 3 Phân tích phương pháp nghiên cứu gốc

### 3.1 Vấn đề và lựa chọn phương pháp

Phân tích mạch lập luận từ chi phí thiết kế đặc trưng thủ công đến lựa chọn CNN, cùng sự phù hợp của các phép toán với tín hiệu. Tách động cơ lý thuyết khỏi bằng chứng thực nghiệm: mô tả tính phân cấp là lý do xây dựng mô hình, còn các bảng chỉ số phản ánh kết quả phân loại trong điều kiện tác giả thử [R1, §1–3].

### 3.2 Các biến và thiết kế khảo sát

Lập bảng theo từng khảo sát: số lớp, số feature maps, kích thước kernel, pooling và learning rate. Với mỗi khảo sát, ghi biến thay đổi, điều kiện giữ lại từ bước trước, chỉ số được xem và kết luận có thể rút ra. Quy trình greedy khiến các khảo sát có quan hệ phụ thuộc; không diễn giải chúng như thiết kế factorial đầy đủ [R1, §3.2.2–4.2].

### 3.3 Bằng chứng và phản biện

Đối chiếu các cấu hình 94,79% và 95,75% bằng Bảng 5. Phân tích Bảng 3–4 để chỉ ra sự khác nhau giữa hoạt động chuyển động và tĩnh. Đề cập thí nghiệm bổ sung thay vì giới hạn phần mô tả bài gốc ở UCI HAR.

Đặt câu hỏi về thông tin validation, chuẩn hóa, mức độ lựa chọn siêu tham số và khả năng khái quát. Khi thông tin chưa đủ, viết rõ chưa xác minh được; không chuyển sự thiếu thông tin thành kết luận có sai lệch hoặc rò rỉ dữ liệu. Dùng `docs/research/original-vs-reproduction.md` làm khung đối chiếu trước khi viết phần thảo luận.

## Nguồn quản lý cho bản thảo

- R1: Ronao & Cho (2016), toàn văn trong `Paper/Human_activity_recognition_with_smartpho.pdf`.
- R2: Anguita et al. (2013), [ESANN](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf), và README UCI HAR trong workspace.
- R3: Ordóñez & Roggen (2016), [toàn văn PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/).
- R4: Oluwalade et al. (2021), `Paper/2103.03836v1.pdf`.
- R5: Zhang et al. (2022), Sensors, 22(4), 1476, [DOI](https://doi.org/10.3390/s22041476). `Paper/har_review.pdf` có số mục khác bản xuất bản; vị trí dẫn chứng hiện tại theo bản local.

Thông tin đầy đủ và vị trí dẫn chứng được quản lý trong `docs/research/literature-matrix.md`. Các ký hiệu R1–R5 là mã nguồn làm việc, sẽ được chuyển sang định dạng trích dẫn thống nhất khi biên tập báo cáo cuối.

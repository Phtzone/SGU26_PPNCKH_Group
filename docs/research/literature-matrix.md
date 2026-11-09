# Bảng dẫn chứng và tổng quan tài liệu

Tài liệu làm việc ngày 09/11/2026. Số trang của Ronao & Cho là số trang in 235–244, khác chỉ số trang PDF 1–10. Các kết quả dưới đây thuộc nghiên cứu được dẫn, không phải kết quả của nhóm.

| Công trình | Câu hỏi / hướng tiếp cận | Dữ liệu và biểu diễn | Bằng chứng đã đọc | Vai trò / giới hạn đối chiếu |
|---|---|---|---|---|
| Ronao & Cho (2016) | CNN học đặc trưng cục bộ, phân cấp từ tín hiệu; khảo sát cấu trúc và siêu tham số | UCI HAR; 128 thời điểm, 6 kênh mô tả; tín hiệu chuẩn hóa | §3–4, Bảng 1–5: CNN+MLP 94,79%; tFFT+CNN 95,75%; HCF+SVM 94,61% | Bài nền; khác mô hình, đầu vào và protocol validation của nhóm |
| Anguita et al. (2013) | Xây dựng dữ liệu công khai cho HAR bằng smartphone và cung cấp thực nghiệm MC-SVM | 30 người, smartphone tại eo; accelerometer/gyroscope; 561 đặc trưng | §2, trang 438–439; §3/Bảng 4, trang 440–441: kết quả MC-SVM được báo cáo làm tròn khoảng 96% | Nguồn gốc dataset; không đồng nhất con số này với SVM 94,61% trong bài Ronao & Cho |
| Ordóñez & Roggen (2016) | Kết hợp trích đặc trưng CNN và mô hình hóa thời gian bằng LSTM | OPPORTUNITY và Skoda; cảm biến đeo đa phương thức | §3: bốn lớp convolution, hai lớp LSTM, không pooling; §5–6 phân tích kết quả và siêu tham số | Bổ sung hướng phát triển; không so trực tiếp điểm số với UCI HAR |
| Oluwalade et al. (2021), bản arXiv v1 trong workspace | So sánh phone/watch, các mô hình CNN/LSTM và dự báo | WISDM; gộp 18 hoạt động thành 15 lớp; 44 đặc trưng được lựa chọn | §3–4, Bảng 1: CNN watch accelerometer Macro-F1 0,849; §4 mô tả split 80/10/10 | Ví dụ về vai trò cảm biến/thiết bị; split chưa được mô tả đủ để khẳng định tách người; không chuyển kết quả sang UCI HAR |
| Zhang et al., bản `har_review.pdf` trong workspace; công trình xuất bản năm 2022 | Tổng quan AE, DBN, CNN, RNN, DRL và GAN cho wearable HAR | Nhiều bộ dữ liệu, cảm biến và ứng dụng | Bản local: §3.3 CNN, §3.4 RNN; §4.1–4.3 bàn dữ liệu, nhãn, phân đoạn và khả năng khái quát | Đã đối chiếu nhan đề/tác giả với bản xuất bản; số mục của bản local khác bản tạp chí |

## Dẫn chứng chi tiết của bài chính

| Mệnh đề cần dùng | Vị trí trong bài | Cách diễn giải trong đồ án |
|---|---|---|
| CNN tận dụng tương quan cục bộ theo thời gian và cấu trúc phân cấp | Abstract, §1, §3.2; trang 235, 237–238 | Giải thích động cơ phương pháp, không coi Accuracy là bằng chứng trực tiếp rằng đặc trưng có ngữ nghĩa cụ thể |
| Đầu vào mô tả có 6 kênh và 128 giá trị mỗi cửa sổ | §4.1/Bảng 1, trang 239 | Công khai khác biệt với đầu vào 9 kênh của nhóm; chưa suy đoán chính xác tệp gia tốc nếu không có mã gốc |
| Dữ liệu có 7.352 train / 2.947 test, 21/9 người | §4.1, trang 239 | Đối chiếu với dataset thực tế; fit-train của nhóm còn 5.982 sau khi dành validation |
| Huấn luyện sử dụng SGD và momentum | §3.2, trang 238; §4.2, trang 239 | Adam của nhóm là lựa chọn triển khai khác bài gốc |
| Khảo sát layer, feature maps, filter, pooling theo thứ tự greedy | §3.2.2–4.2, trang 239; Hình 4–6 | Phân biệt quy trình khảo sát rộng của tác giả với E2 ba cấu hình của nhóm |
| Mô hình tốt 96–192–192–1000–6, kernel 9, pooling 3, learning rate 0,02 | §4.2, trang 240 | Gắn đúng cấu hình với Accuracy 94,79% |
| tFFT đi cùng CNN một lớp có 200 feature maps | Bảng 5 và đoạn văn, trang 242 | Accuracy 95,75% không phải của mọi CNN hoặc của mô hình ba lớp phía trên |
| CNN yếu hơn ở nhóm hoạt động tĩnh trong thí nghiệm tác giả | Bảng 3–4 và thảo luận, trang 241–242 | Recall nằm 87,71%, ngồi 88,80%, đứng 93,23%; không thu hẹp thành chỉ cặp ngồi–đứng |
| Có thực nghiệm bổ sung | §4.2 Additional experiment, trang 242–244 | Có dữ liệu 3 người, 5 hoạt động; không mô tả tác giả chỉ thực nghiệm trên một dataset |
| Có chọn checkpoint theo validation | §4.2, trang 239 | Chưa đủ thông tin để khẳng định cách tạo validation hoặc cách fit chuẩn hóa giống nhóm |

## Nguồn truy nguyên

1. Ronao, C. A., & Cho, S.-B. (2016). *Human activity recognition with smartphone sensors using deep learning neural networks*. Expert Systems with Applications, 59, 235–244. [DOI](https://doi.org/10.1016/j.eswa.2016.04.032). Toàn văn: `Paper/Human_activity_recognition_with_smartpho.pdf`.
2. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN, 437–442. [Toàn văn từ ESANN](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf), đã đọc ngày 09/11/2026.
3. Ordóñez, F. J., & Roggen, D. (2016). *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. Sensors, 16(1), 115. [Toàn văn PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/), [DOI](https://doi.org/10.3390/s16010115), đã đọc ngày 09/11/2026.
4. Oluwalade, B., Neela, S., Wawira, J., Adejumo, T., & Purkayastha, S. (2021). *Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data*. Bản v1 trong `Paper/2103.03836v1.pdf`, có đầu trang HEALTHINF 2021.
5. Zhang, S., Li, Y., Zhang, S., Shahabi, F., Xia, S., Deng, Y., & Alshurafa, N. (2022). *Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances*. Sensors, 22(4), 1476. [Bản xuất bản PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8879042/), [DOI](https://doi.org/10.3390/s22041476); metadata đối chiếu ngày 09/11/2026. Có [tiền ấn bản arXiv 2111.00418](https://arxiv.org/abs/2111.00418) từ 2021. `Paper/har_review.pdf` là bản local có cách đánh số mục khác, vì vậy khi dẫn vị trí phải nói rõ phiên bản, không chuyển nguyên số mục sang bản tạp chí.
6. `Dataset/UCI HAR Dataset/README.txt` và `features_info.txt`: thông tin phiên bản dữ liệu, kênh, nhãn và tiền xử lý. Chín chuỗi dữ liệu không đồng nghĩa có chín trục cảm biến độc lập: tổng gia tốc và gia tốc cơ thể là các biểu diễn liên quan.


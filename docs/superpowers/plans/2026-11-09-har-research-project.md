# Kế hoạch triển khai đồ án HAR trong một tuần

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking. Kế hoạch này là đầu ra của bước chuẩn bị; các notebook, kết quả, báo cáo và slide bên dưới là sản phẩm cần thực hiện tiếp, chưa được tạo.

**Goal:** Hoàn thành đồ án môn Phương pháp nghiên cứu khoa học trong Khoa học dữ liệu: phân tích Ronao & Cho (2016), thực nghiệm CNN trên UCI HAR, viết báo cáo và chuẩn bị thuyết trình trong khoảng một tuần.

**Architecture:** Giữ bộ dữ liệu và tài liệu gốc; triển khai một notebook chính cho kiểm tra dữ liệu, chia tập, huấn luyện và đánh giá. Lưu cấu hình, lịch sử huấn luyện, dự đoán và bảng kết quả thành tệp độc lập để truy nguyên các số liệu trong báo cáo. Viết phần học thuật song song với thực nghiệm.

**Tech Stack:** Python, NumPy, pandas, TensorFlow/Keras, scikit-learn, matplotlib, Jupyter Notebook hoặc Google Colab theo đề cương. Phiên bản cụ thể được kiểm tra và ghi lại sau khi môi trường chạy thử thành công; chưa giả định workspace đã có môi trường huấn luyện phù hợp.

**Spec:** [Đề cương hiện tại](../../../De_cuong_NCKH_HAR_CNN_Ronao_Cho.docx), [bài Ronao & Cho](../../../Paper/Human_activity_recognition_with_smartpho.pdf), [mô tả UCI HAR](../../../Dataset/UCI%20HAR%20Dataset/README.txt), cùng giới hạn thời gian khoảng một tuần do người dùng cung cấp.

Ngày lập: 09/11/2026. Ngày 1–7 bên dưới là ngày làm việc tương đối, không phải hạn nộp đã được giảng viên xác nhận.

Tiến độ cập nhật: xem [nhật ký triển khai](../../research/progress.md). Đợt khởi động đã hoàn thành pipeline dữ liệu, notebook và ba lượt CNN-B trên validation; chưa thực hiện đánh giá test.

## 1. Phạm vi và thứ tự ưu tiên

Tên đề tài làm việc: **Nghiên cứu và tái hiện một phần mô hình mạng nơ-ron tích chập trong nhận dạng hoạt động con người từ dữ liệu cảm biến điện thoại thông minh**.

| Phương án | Công việc | Quyết định trong một tuần |
|---|---|---|
| Tối thiểu đầy đủ | Phân tích bài gốc + CNN cơ sở + baseline lớp phổ biến + phân tích lỗi | Bắt buộc, ưu tiên hoàn thành trước |
| Mở rộng vừa sức | Thêm CNN một và ba lớp tích chập để đối chiếu với CNN cơ sở | Thực hiện nếu ngày 3 còn đủ thời gian và tài nguyên |
| Mở rộng rộng hơn | SVM, thử 6 so với 9 kênh, FFT, CNN–LSTM hoặc dữ liệu mới | Để sau bài nộp chính |

Đóng góp của đồ án là cách đặt câu hỏi, thiết kế thực nghiệm, kiểm soát sai lệch, tái lập và phản biện. Không đặt điều kiện thành công là đạt Accuracy 94,79% hoặc 95,75%.

### Quy tắc áp dụng toàn bộ kế hoạch

- Phân biệt nội dung bài gốc, thiết kế của nhóm và kết quả nhóm thực sự thu được.
- Giữ nguyên 2.947 mẫu test chính thức; không dùng test để chọn cấu hình, số epoch hoặc seed.
- Chia train/validation theo người tham gia; chỉ fit chuẩn hóa trên phần dùng để huấn luyện.
- CNN chính dùng tín hiệu dạng `(N, 128, 9)`; 561 đặc trưng tính sẵn là một biểu diễn khác.
- Gọi công việc là tái hiện một phần. Bài gốc mô tả đầu vào 6 kênh; thiết kế 9 kênh là khác biệt cần công khai.
- Giữ lại tất cả lần chạy hợp lệ theo các seed đã định; không chỉ trình bày lần có kết quả tốt nhất.
- Mọi bảng và hình thực nghiệm phải có tệp nguồn tương ứng. Không tạo số liệu để lấp chỗ trống trong báo cáo.
- Khoảng 25–30 trang là quy mô đề xuất trong đề cương, chưa phải yêu cầu chính thức đã được xác nhận. Rubric và hướng dẫn của giảng viên được ưu tiên khi có.

## 2. Hiện trạng đã kiểm tra

| Thành phần | Trạng thái |
|---|---|
| Đề cương | Đã có cấu trúc sáu chương, câu hỏi, giả thuyết và thiết kế CNN sơ bộ |
| Bài chính | Đã có toàn văn Ronao & Cho (2016), 10 trang |
| Tài liệu phụ | Đã có bài tổng quan HAR và nghiên cứu WISDM trên smartphone/smartwatch |
| UCI HAR | Đủ 7.352 mẫu train, 2.947 mẫu test, sáu lớp |
| Tín hiệu | Mỗi tập có chín tệp, mỗi dòng 128 giá trị; đã kiểm tra kích thước và giá trị hữu hạn |
| Đặc trưng tính sẵn | Train `(7352, 561)`; test `(2947, 561)` |
| Người tham gia | 21 người train, 9 người test; không giao nhau |
| Mã và kết quả | Chưa có notebook, mô hình đã huấn luyện, lịch sử chạy hoặc kết quả thực nghiệm |
| Yêu cầu nộp | Chưa có rubric, thời lượng trình bày, danh sách thành viên hoặc hạn nộp chính xác trong workspace |

Các điểm đã đối chiếu trong bài gốc:

- Bảng 1, trang in 239: đầu vào dài 128, có 6 kênh; thiết lập huấn luyện khác CNN đề xuất.
- Trang in 240: cấu hình 96–192–192–1000–6, kernel 9, pooling 3, learning rate 0,02 đạt 94,79%.
- Bảng 5, trang in 242: 95,75% thuộc cấu hình bổ sung tFFT; không phải cùng cấu hình chỉ đổi tên.
- Bảng 3, trang in 242: Recall lớp nằm là 87,71%, thấp nhất trong bảng; ngồi là 88,80%, đứng là 93,23%.
- Bài gốc có thí nghiệm bổ sung trên dữ liệu của ba người; cần đề cập khi phân tích phạm vi bằng chứng, dù nhóm chỉ thực nghiệm trên UCI HAR.
- Bài gốc nhắc validation nhưng phần mô tả chưa đủ để khẳng định cách tách theo người giống thiết kế của nhóm. Khi phản biện, phân biệt thông tin thiếu với bằng chứng về sai sót.

## 3. Lịch bảy ngày và điểm dừng

Ước lượng khoảng 24–30 giờ làm việc chủ động cho toàn nhóm, ngoài thời gian máy huấn luyện. Đây là mức dự kiến để phân bổ việc; điều chỉnh theo tốc độ viết và kết quả chạy thử ngày 2.

| Ngày | Thời lượng dự kiến | Việc chính | Đầu ra kiểm tra cuối ngày |
|---|---|---|---|
| 1 | 3–4 giờ | Xác nhận yêu cầu môn; chốt phạm vi; lập bảng đọc bài; chuẩn bị môi trường | Protocol bản đầu, bảng dẫn chứng bài gốc, dàn ý sáu chương |
| 2 | 4–5 giờ | Đọc dữ liệu, chia theo người, chuẩn hóa, kiểm tra pipeline, chạy CNN thử một epoch | Notebook chạy được từ đầu đến cuối ở chế độ thử; bảng phân bố lớp; hình tín hiệu |
| 3 | 3–4 giờ + thời gian máy | Huấn luyện CNN-B theo ba seed; nếu ổn thì chạy CNN-A/C | Cấu hình và lịch sử chạy được lưu; có quyết định giữ hoặc bỏ E2 |
| 4 | 3–4 giờ | Khóa cấu hình; đánh giá test; xuất bảng, hình và dự đoán; viết phần kết quả | Bộ kết quả có nguồn; bản nháp Chương 4–5 |
| 5 | 5–6 giờ | Hoàn thiện sáu chương, trích dẫn, bảng đối chiếu và phần hạn chế | Báo cáo đủ nội dung, không còn kết quả giả định |
| 6 | 3–4 giờ | Rà soát báo cáo; chuẩn bị slide; diễn tập và chuẩn bị trả lời phản biện | Báo cáo gần cuối, slide và ghi chú trình bày |
| 7 | 2–3 giờ | Kiểm tra khả năng chạy lại, hình/bảng, tài liệu tham khảo và gói nộp | Bộ nộp cuối cùng, có README hướng dẫn |

Viết Chương 1–3 từ ngày 1 đến ngày 3 trong lúc chờ huấn luyện. Không dồn toàn bộ phần học thuật sang ngày 5.

**Điểm cắt phạm vi:** Cuối ngày 3, nếu CNN-B chưa hoàn thành ổn định hoặc việc viết bị chậm, bỏ E2 khỏi mục tiêu bắt buộc. Nếu chỉ đủ một lần chạy CNN-B, báo cáo đúng một seed và ghi giới hạn về độ ổn định; không tạo độ lệch chuẩn từ một lần chạy. Ngày 5 trở đi ưu tiên bài nộp, không mở thêm nhánh mô hình.

## 4. Câu hỏi nghiên cứu và bằng chứng cần thu

| Mã | Câu hỏi / giả thuyết | Bằng chứng | Phạm vi |
|---|---|---|---|
| RQ1 | Vì sao CNN phù hợp với chuỗi tín hiệu cảm biến? | Phân tích tính cục bộ theo thời gian, chia sẻ trọng số, pooling và học đặc trưng trong bài gốc | Bắt buộc, phần học thuật |
| RQ2 / H1 | CNN-B nhận dạng sáu lớp ra sao, có vượt dự đoán lớp phổ biến nhất không? | Accuracy, Macro-F1, Precision/Recall/F1 từng lớp của CNN-B và baseline trên cùng test | Bắt buộc, E1 |
| RQ3 / H2 | Các hoạt động nào bị nhầm nhiều; ngồi–đứng có thực sự nổi bật không? | Confusion matrix, số lỗi theo cặp hai chiều, Recall từng lớp | Bắt buộc, E1 |
| RQ4 / H3 | Ba cấu hình có độ sâu khác nhau cho kết quả và chi phí khác nhau như thế nào? | Trung bình, độ lệch chuẩn qua seed; số tham số; thời gian chạy cùng điều kiện | Có điều kiện, E2 |
| RQ5 | Kết quả nhóm có thể đối chiếu đến đâu với Ronao & Cho? | Bảng đối chiếu đầu vào, kiến trúc, tối ưu hóa, chia tập, chỉ số và giới hạn | Bắt buộc |

H4 về khả năng cạnh tranh với SVM không được kết luận trong bài nộp nếu không thực hiện E3. Nếu bỏ E2, RQ4/H3 được chuyển thành hướng mở rộng, không trình bày như giả thuyết đã kiểm chứng.

## 5. Protocol thực nghiệm đề xuất

Các lựa chọn dưới đây được ghi trước khi huấn luyện nhằm tránh thay đổi theo kết quả test. Đây là thiết kế của nhóm, không phải thông số nguyên bản của bài báo.

### 5.1. Đầu vào và chia tập

Thứ tự kênh cố định: `body_acc_x`, `body_acc_y`, `body_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z`, `total_acc_x`, `total_acc_y`, `total_acc_z`.

Ghép chín tệp theo chiều cuối thành `(N, 128, 9)`, giữ nguyên thứ tự dòng để khớp `y` và `subject`. Chuyển nhãn 1–6 thành 0–5 cho loss; giữ bảng tên hoạt động khi xuất kết quả. Dùng `float32` cho đầu vào mô hình.

Đề xuất validation cố định gồm người **5, 11, 19, 26**. Danh sách được chọn trước khi chạy mô hình, không dựa vào độ khó hoặc kết quả dự đoán. Đã kiểm tra trực tiếp rằng hai phần có đủ sáu lớp.

| Tập | Số người | Số cửa sổ | Mục đích |
|---|---:|---:|---|
| Fit-train | 17 | 5.982 | Fit chuẩn hóa, học trọng số và xác định lớp phổ biến |
| Validation | 4 | 1.370 | Early stopping và đánh giá cấu hình trong giai đoạn phát triển |
| Test chính thức | 9 | 2.947 | Đánh giá cuối sau khi khóa protocol |

| Nhãn gốc | Hoạt động | Fit-train | Validation | Test |
|---:|---|---:|---:|---:|
| 1 | Đi bộ | 1.000 | 226 | 496 |
| 2 | Đi lên cầu thang | 877 | 196 | 471 |
| 3 | Đi xuống cầu thang | 804 | 182 | 420 |
| 4 | Ngồi | 1.038 | 248 | 491 |
| 5 | Đứng | 1.124 | 250 | 532 |
| 6 | Nằm | 1.139 | 268 | 537 |

Chuẩn hóa từng kênh: tính mean và standard deviation trên trục mẫu và thời gian của **fit-train**, sau đó áp dụng cùng tham số cho validation/test. Thay standard deviation nhỏ hơn `1e-8` bằng `1e-8`. Lưu mean/std để có thể áp dụng lại. Tín hiệu đã được bộ dữ liệu tiền xử lý và phân cửa sổ; không mô tả chúng là dữ liệu hoàn toàn chưa xử lý.

Để giảm công việc và tránh thay đổi protocol trong tuần cuối, các mô hình cuối được giữ ở chế độ huấn luyện trên 5.982 mẫu; không thêm bước huấn luyện lại trên toàn bộ 7.352 mẫu. Khi đối chiếu bài gốc, ghi rõ khác biệt số mẫu thực sự dùng để cập nhật trọng số.

### 5.2. Baseline và CNN

**Baseline bắt buộc:** Dự đoán lớp xuất hiện nhiều nhất trong fit-train, xác định bằng nhãn train. Theo phép chia này đó là lớp nằm, 1.139/5.982 mẫu. Baseline không được chọn dựa trên phân bố test. Tính cùng bộ chỉ số như CNN, dùng `zero_division=0` cho Precision/F1 của lớp không được dự đoán.

**CNN-B cơ sở:** Input `(128,9)` → Conv1D 64 filters, kernel 5, ReLU, padding `same` → MaxPooling1D size 2 → Conv1D 128 filters, kernel 3, ReLU, padding `same` → GlobalAveragePooling1D → Dense 64 ReLU → Dropout 0,3 → Dense 6 Softmax.

| Tham số | Giá trị khóa ban đầu |
|---|---|
| Optimizer | Adam |
| Learning rate | 0,001 |
| Batch size | 64 |
| Epoch tối đa | 50 |
| Loss | Sparse categorical cross-entropy |
| Early stopping | `val_loss`, patience 8, khôi phục trọng số tốt nhất |
| Checkpoint | Lưu trọng số tại `val_loss` thấp nhất, không lưu mặc định epoch cuối |
| Seed | 42, 123, 2026 |
| Tập validation | Cố định cùng bốn người cho mọi seed và cấu hình |
| Xáo trộn | Chỉ xáo trộn các mẫu trong fit-train khi huấn luyện |

Một lượt chạy thử một epoch nhằm kiểm tra pipeline phải được gắn nhãn `smoke`, không đưa vào bảng kết quả chính. Khi thay đổi kiến trúc hoặc siêu tham số sau chạy thử, cập nhật protocol trước khi chạy chính thức và ghi lý do.

### 5.3. E2 có điều kiện

| Cấu hình | Phần tích chập | Phần cuối |
|---|---|---|
| CNN-A | Conv 64/kernel 5 → Pool 2 | GAP → Dense 64 → Dropout 0,3 → Softmax 6 |
| CNN-B | Conv 64/kernel 5 → Pool 2 → Conv 128/kernel 3 | Giống CNN-A |
| CNN-C | Conv 64/kernel 5 → Pool 2 → Conv 128/kernel 3 → Conv 128/kernel 3 | Giống CNN-A |

Mọi Conv dùng ReLU và padding `same`. Giữ nguyên split, chuẩn hóa, optimizer, learning rate, batch size, giới hạn epoch và seed. E1 gồm ba lượt CNN-B; E2 bổ sung sáu lượt CNN-A/C, tổng chín lượt, không huấn luyện lại CNN-B chỉ vì chuyển sang E2.

Số lớp, số tham số và receptive field cùng thay đổi, nên kết luận là so sánh **ba cấu hình có độ sâu khác nhau**, không khẳng định đã cô lập hoàn toàn tác động nhân quả của độ sâu.

Nếu cần chọn một cấu hình đại diện, dùng trung bình Macro-F1 validation qua ba seed; khi hòa đến bốn chữ số thập phân, chọn mô hình ít tham số hơn. Vẫn báo cáo các cấu hình đã chạy theo kế hoạch, không chỉ giữ cấu hình thắng. Nếu chưa đủ ba seed cho các cấu hình mở rộng, ưu tiên bỏ E2 khỏi kết luận chính hoặc ghi rõ đây là khảo sát sơ bộ với số lượt không cân bằng.

### 5.4. Đánh giá và lưu bằng chứng

- Đến ngày 4, chốt danh sách cấu hình, seed và checkpoint trước khi tính các chỉ số test.
- Xuất Accuracy, Macro-F1, Precision/Recall/F1/support từng lớp cho mỗi lượt chạy.
- Khi có ba seed, báo cáo trung bình và độ lệch chuẩn mẫu (`ddof=1`), đồng thời giữ bảng từng seed.
- Lưu dự đoán gồm `sample_index`, `subject_id`, `y_true`, `y_pred` và sáu xác suất lớp; giữ index theo thứ tự test gốc.
- Xuất confusion matrix đếm mẫu và chuẩn hóa theo hàng. Dùng CNN-B seed 42 làm hình minh họa đã định trước; nếu E2 có mô hình khác được chọn, có thể bổ sung hình seed 42 của mô hình đó và ghi rõ.
- Đối chiếu lỗi giữa các seed; nếu cặp nhầm nổi bật thay đổi, nêu sự thay đổi thay vì khẳng định một cặp cố định.
- Không cộng confusion matrix của ba seed rồi trình bày như thể có 8.841 mẫu test độc lập.
- Ghi thời gian huấn luyện, số epoch thực chạy, số tham số, phần cứng và phiên bản thư viện. Chỉ so sánh chi phí trực tiếp khi điều kiện phần cứng phù hợp.
- Độ lệch chuẩn qua seed phản ánh biến động huấn luyện trên một split, không phải khoảng tin cậy về toàn bộ người dùng hoặc bằng chứng có ý nghĩa thống kê.
- Sau khi xem test, mọi thay đổi dựa vào kết quả đó phải ghi là phân tích khám phá; không tiếp tục gọi test này là tập đánh giá chưa từng được sử dụng.

## 6. Tệp và sản phẩm dự kiến

Chỉ tạo tệp khi thực hiện công việc tương ứng; không tạo trước bảng số liệu hoặc mô hình rỗng mang tên kết quả.

| Tệp / thư mục | Trách nhiệm |
|---|---|
| `docs/research/protocol.md` | Câu hỏi, split, cấu hình, seed, chỉ số và lịch sử thay đổi quyết định |
| `docs/research/literature-matrix.md` | Bảng công trình, dữ liệu, phương pháp, bằng chứng, hạn chế, vị trí nguồn |
| `docs/research/original-vs-reproduction.md` | Đối chiếu bài gốc và thiết kế nhóm |
| `notebooks/01_har_experiment.ipynb` | Notebook chính, chia mục rõ ràng từ đọc dữ liệu đến xuất kết quả |
| `requirements.txt` | Phiên bản các thư viện đã chạy thành công trong môi trường thực tế |
| `results/data_summary.json` | Shape, phân bố lớp, người tham gia và các kiểm tra dữ liệu |
| `results/split.json` | ID người tham gia và index mẫu của fit-train/validation/test |
| `results/normalization.npz` | Mean/std theo chín kênh, chỉ fit trên fit-train |
| `results/runs/<model>_seed<seed>/` | `config.json`, `environment.json`, `history.csv`, `validation_metrics.json`, `test_metrics.json`, `test_predictions.csv`, `model.keras` |
| `results/metrics_per_run.csv` | Một dòng mỗi lượt chạy hợp lệ, ghi rõ split đánh giá |
| `results/metrics_summary.csv` | Trung bình, độ lệch chuẩn, số lượt của từng cấu hình |
| `results/figures/` | Phân bố lớp, tín hiệu mẫu, learning curves, confusion matrix và hình so sánh nếu có E2 |
| `report/bao_cao_har.docx` | Bản báo cáo chính có thể chỉnh sửa |
| `report/bao_cao_har.pdf` | Bản xuất để kiểm tra hoặc nộp nếu được yêu cầu |
| `slides/thuyet_trinh_har.pptx` | Slide dựa trên nội dung và kết quả đã chốt |
| `README.md` | Mục tiêu, đường dẫn dữ liệu, cách mở notebook, chạy thử, chạy chính thức và đọc kết quả |

`<model>` và `<seed>` trong đường dẫn là quy tắc đặt tên, ví dụ `results/runs/cnn_b_seed42/`. Nếu thay đổi cấu hình, thêm mã phiên bản để tránh ghi đè một lượt chạy đã dùng trong báo cáo.

## 7. Các đầu việc thực hiện

### M1 — Chốt phạm vi và yêu cầu nộp, ngày 1

**Tệp:** `docs/research/protocol.md`; dàn ý trong bản báo cáo làm việc.

- [ ] Ghi hạn nộp chính xác, định dạng nộp, thời lượng trình bày và rubric khi nhận được hướng dẫn giảng viên.
- [ ] Gán người phụ trách ba vai trò: tổng quan/phương pháp, thực nghiệm, biên tập/thuyết trình. Nếu làm một người thì giữ cùng thứ tự nhiệm vụ.
- [x] Chốt mục tiêu tối thiểu E1; đánh dấu E2 có điều kiện, H4 ngoài phạm vi chính.
- [x] Ghi câu hỏi nghiên cứu, giả thuyết và giới hạn phát biểu kết luận vào protocol.
- [x] Ghi split, thứ tự kênh, cấu hình CNN-B, seed và thước đo như Mục 5.

**Hoàn thành khi:** Người đọc biết chính xác nhóm sẽ kiểm chứng điều gì, bằng dữ liệu nào, và bằng chứng nào cần có. Không còn mục tiêu thực nghiệm nằm ngoài lịch một tuần.

### M2 — Lập bằng chứng học thuật, ngày 1–2

**Tệp:** `docs/research/literature-matrix.md`, `docs/research/original-vs-reproduction.md`.

- [x] Đọc lại bài chính theo các mục: vấn đề, lý do dùng CNN, thiết lập, biến khảo sát, kết quả, giới hạn.
- [x] Lập bảng có các cột: công trình, câu hỏi, dữ liệu, biểu diễn đầu vào, mô hình, cách đánh giá, kết quả được dẫn, hạn chế, trang/bảng nguồn.
- [x] Ghi các số liệu 94,79%, 95,75%, 94,61% đúng cấu hình và đúng Bảng 5; không dùng số của bài WISDM làm kết quả trên UCI HAR.
- [x] Đối chiếu sáu so với chín kênh, kiến trúc, optimizer, epoch, split validation và số mẫu huấn luyện thực tế.
- [x] Tìm và đọc nguồn gốc Anguita et al. (2013) nếu đưa mô tả nghiên cứu tạo dữ liệu vào tổng quan; đọc toàn văn Ordóñez & Roggen (2016) nếu bổ sung DeepConvLSTM như một công trình được phân tích trực tiếp. Đã truy cập nguồn ESANN/PMC, liên kết trong bảng dẫn chứng; chưa lưu hai PDF mới vào workspace.
- [x] Viết nháp Chương 1 và khung Chương 2–3 bằng lời của nhóm, gắn nguồn cho các khẳng định học thuật.

**Hoàn thành khi:** Mỗi khẳng định quan trọng có nguồn truy nguyên; phân biệt được nhận xét của nhóm và kết luận của tác giả. Ưu tiên 4–5 nguồn thực sự đọc thay vì mở rộng danh mục nhưng thiếu kiểm chứng.

### M3 — Hoàn thành pipeline dữ liệu, ngày 2

**Tệp:** `notebooks/01_har_experiment.ipynb`, `requirements.txt`, `results/data_summary.json`, `results/split.json`, `results/normalization.npz`.

- [x] Chọn một môi trường huấn luyện; kiểm tra import thư viện và ghi phần cứng/phiên bản. Không tự giả định Python hiện có trên máy đủ để huấn luyện TensorFlow.
- [x] Viết các mục notebook: cấu hình → đọc dữ liệu → kiểm tra → split → chuẩn hóa → mô hình → huấn luyện → đánh giá validation → xuất kết quả. Đánh giá test được bổ sung sau khi khóa protocol ở M5.
- [x] Đọc và ghép chín kênh theo thứ tự đã chốt; đọc nhãn và subject; kiểm tra shape, số dòng, sáu lớp và NaN/Inf.
- [x] Tách fit-train/validation theo danh sách người; lưu index; xác nhận ba tập người không giao nhau.
- [x] Fit chuẩn hóa riêng trên fit-train, lưu và áp dụng cho các tập còn lại.
- [x] Xuất bảng phân bố lớp và hình tín hiệu mẫu từ dữ liệu thật, có nhãn trục thời gian và đơn vị phù hợp.
- [x] Chạy lại notebook từ kernel sạch đến hết phần dữ liệu để kiểm tra không phụ thuộc biến của lần chạy trước.

**Kiểm tra cần có:** Các shape, số mẫu và phân bố lớp khớp Mục 5; phép ghép giữ đúng kênh; index giữ đúng nhãn/subject. Với một mảng nhỏ kiểm soát được, việc thay đổi validation/test không làm đổi mean/std đã fit trên train. Đây là kiểm tra hành vi chống rò rỉ dữ liệu, không chỉ là kiểm tra code chạy không lỗi.

**Hoàn thành khi:** Có đầu vào hợp lệ `(5982,128,9)`, `(1370,128,9)`, `(2947,128,9)`, nhãn 0–5 và bằng chứng chia tập không trùng người.

### M4 — Huấn luyện và kiểm tra CNN-B, ngày 2–3

**Tệp:** Notebook và `results/runs/cnn_b_seed42/`, `cnn_b_seed123/`, `cnn_b_seed2026/`.

- [x] Tạo baseline từ nhãn fit-train; trước khi khóa đánh giá chỉ kiểm tra chỉ số trên validation.
- [x] Xây CNN-B đúng cấu hình; kiểm tra output có sáu xác suất, tổng mỗi hàng gần 1 và loss hữu hạn.
- [x] Chạy thử một epoch, lưu riêng kết quả `smoke`; ước lượng thời gian hoàn thành ba seed từ tốc độ thực tế.
- [x] Đặt seed cho các nguồn ngẫu nhiên; bật chế độ quyết định được khi môi trường hỗ trợ và ghi giới hạn nếu có.
- [x] Huấn luyện ba seed, tái khởi tạo mô hình và optimizer cho từng lượt; lưu history, config, environment và checkpoint tốt nhất theo validation loss.
- [x] Mở lại mô hình đã lưu và kiểm tra dự đoán validation khớp trong sai số số học hợp lý.
- [x] Xem learning curves, kiểm tra loss và phân bố dự đoán; nếu có lỗi dữ liệu hoặc code, ghi nhận và sửa trước khi chạy lại các lượt bị ảnh hưởng.

**Hoàn thành khi:** Có ít nhất một lượt CNN-B hợp lệ với đủ bằng chứng lưu lại; mục tiêu là ba lượt. Chưa tính điểm test để quyết định có đổi mô hình hay không.

### M5 — Quyết định E2 và khóa protocol, cuối ngày 3

**Tệp:** Protocol, notebook, các thư mục lượt chạy CNN-A/C nếu thực hiện.

- [ ] Đánh giá tiến độ viết và thời gian huấn luyện từ M4.
- [ ] Nếu đủ điều kiện, tạo CNN-A/C theo Mục 5.3 và chạy cùng ba seed; không đổi split hoặc các tham số chung.
- [ ] Nếu không đủ điều kiện, ghi E2 không thực hiện và cập nhật mục tiêu, giả thuyết cùng dàn ý báo cáo.
- [ ] Tổng hợp chỉ số validation và ghi cấu hình đại diện theo quy tắc đã định.
- [ ] Lưu bản protocol khóa, danh sách các lượt hợp lệ và những lần chạy lỗi bị loại kèm lý do.

**Hoàn thành khi:** Danh sách mô hình/seed để đánh giá cuối đã cố định trước khi xem kết quả test.

### M6 — Đánh giá cuối và xuất kết quả, ngày 4

**Tệp:** Các tệp test trong thư mục run, `results/metrics_per_run.csv`, `results/metrics_summary.csv`, `results/figures/`.

- [ ] Nạp từng checkpoint đã khóa; đánh giá trên cùng test chính thức và baseline đã xác định từ train.
- [ ] Xuất Accuracy, Macro-F1, classification report và dự đoán từng mẫu.
- [ ] Tính lại Accuracy/Macro-F1 từ tệp dự đoán đã lưu và đối chiếu với bảng metrics.
- [ ] Kiểm tra tổng phần tử confusion matrix bằng 2.947; tổng từng hàng khớp số mẫu lớp tương ứng.
- [ ] Tổng hợp từng seed rồi tính mean/std nếu có nhiều seed; ghi rõ số lượt trong bảng.
- [ ] Vẽ learning curves, confusion matrix và hình so sánh cấu hình nếu có E2. Ghi rõ split, seed và mô hình trong chú thích.
- [ ] Phân tích các lỗi nổi bật và đánh giá H1/H2, H3 nếu có đủ dữ liệu; ghi rõ khi bằng chứng không ủng hộ giả thuyết.

**Hoàn thành khi:** Mỗi số trong bảng chính có thể tính lại từ tệp kết quả; mỗi hình có chú thích và nguồn dữ liệu rõ ràng.

### M7 — Hoàn thiện báo cáo, ngày 4–5

**Tệp:** `report/bao_cao_har.docx`; bản PDF sau khi kiểm tra.

| Chương | Nội dung cần hoàn thiện | Dung lượng tham khảo |
|---|---|---|
| 1 | Bối cảnh, vấn đề, mục tiêu, câu hỏi, giả thuyết, phạm vi | 3 trang |
| 2 | HAR, cảm biến, UCI HAR, CNN, tổng quan nghiên cứu liên quan | 4–5 trang |
| 3 | Phân tích phương pháp và bằng chứng của Ronao & Cho | 5 trang |
| 4 | Protocol nhóm: split, chuẩn hóa, mô hình, seed, chỉ số, tái lập | 4–5 trang |
| 5 | Kết quả thật, giả thuyết, đối chiếu, sai lệch và giới hạn | 5–6 trang |
| 6 | Kết luận đúng phạm vi, bài học phương pháp và hướng phát triển | 2 trang |

Tổng khoảng 23–26 trang theo phân bổ này; nếu cần theo quy mô 25–30 trang của đề cương, bổ sung phân tích có bằng chứng ở Chương 3 và 5. Không tăng dung lượng bằng lặp lại định nghĩa hoặc kéo giãn trình bày. Mục tiêu cuối cùng theo yêu cầu giảng viên.

- [ ] Viết phần kết quả trực tiếp từ bảng/hình đã xuất; thống nhất tỷ lệ phần trăm và số thập phân.
- [ ] Nêu rõ 6/9 kênh, 5.982 mẫu fit-train, kiến trúc giản lược, optimizer và huấn luyện khác bài gốc.
- [ ] Khi kết quả cao hoặc thấp hơn bài gốc, giải thích điều kiện so sánh; không kết luận vượt trội chỉ từ chênh lệch Accuracy.
- [ ] Nêu hạn chế: một dataset, ít người, thiết bị/vị trí đeo cố định, một split validation, số seed hữu hạn và phạm vi sáu hoạt động.
- [ ] Kiểm tra mọi trích dẫn trong nội dung có mục tương ứng ở danh mục tham khảo và ngược lại.
- [ ] Kiểm tra mỗi hình/bảng được dẫn trong văn bản, có tiêu đề, đơn vị và nguồn.
- [ ] Kết luận chỉ trả lời câu hỏi đã nghiên cứu; bỏ các phát biểu khẳng định giả thuyết chưa được kiểm chứng.

**Hoàn thành khi:** Báo cáo đủ sáu chương, các số liệu thống nhất với kết quả máy, phân biệt rõ bài gốc và công việc của nhóm.

### M8 — Chuẩn bị thuyết trình và nộp, ngày 6–7

**Tệp:** `slides/thuyet_trinh_har.pptx`, báo cáo cuối, notebook và README.

- [ ] Chuẩn bị 10–12 slide theo mạch: vấn đề → bài gốc → câu hỏi nhóm → dữ liệu/split → CNN → protocol → kết quả → lỗi → đối chiếu → hạn chế/kết luận. Điều chỉnh theo thời lượng giảng viên quy định.
- [ ] Chuẩn bị câu trả lời: vì sao chọn bài này; vì sao chia theo người; 6 và 9 kênh khác gì; vì sao dùng Macro-F1; baseline có ý nghĩa gì; có thể kết luận gì nếu kết quả thấp hơn bài gốc.
- [ ] Diễn tập với thời gian thực, bảo đảm mọi thành viên hiểu phương pháp và phần việc của mình.
- [ ] Chạy lại notebook từ kernel sạch ở chế độ kiểm tra dữ liệu/đọc mô hình; không cần huấn luyện lại toàn bộ chỉ để đóng gói nếu cấu hình không đổi.
- [ ] Mở bản báo cáo và slide đã xuất, kiểm tra hình, bảng, công thức, dấu tiếng Việt và ngắt trang.
- [ ] Ghi hướng dẫn đường dẫn dữ liệu, môi trường và thứ tự chạy trong README; tránh đường dẫn tuyệt đối chỉ tồn tại trên máy một thành viên.
- [ ] Kiểm tra yêu cầu có cần nộp trọng số/dataset không để tránh gói nộp quá lớn; giữ bản kết quả gốc trong workspace.
- [ ] Đóng gói đúng yêu cầu môn học và kiểm tra các tệp mở được trước khi nộp.

**Hoàn thành khi:** Bộ nộp mở được, notebook có hướng dẫn, kết quả truy nguyên được, báo cáo/slide nhất quán và nhóm có thể giải thích các quyết định nghiên cứu.

## 8. Buổi làm việc đầu tiên

Trong khoảng 90–120 phút đầu, chỉ cần đạt bốn đầu ra:

1. Ghi yêu cầu nộp đã biết và giao người phụ trách; dùng phạm vi E1 làm cam kết tối thiểu.
2. Tạo protocol từ các quyết định ở Mục 4–5, ghi rõ đây là thiết kế của nhóm.
3. Điền bảng dẫn chứng bài Ronao & Cho bằng các vị trí nguồn đã xác định ở Mục 2.
4. Mở môi trường notebook, kiểm tra thư viện và đọc được một tệp tín hiệu cùng nhãn trước khi chuyển sang M3.

Sau buổi này, công việc tiếp theo là hoàn thành pipeline dữ liệu và chạy thử CNN-B. Không cần chờ đọc hết mọi nghiên cứu liên quan mới bắt đầu thực nghiệm.

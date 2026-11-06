\*\*Ngày chạy:\*\* 06/11/2026



\*\*Runtime Colab:\*\* CPU



\*\*Python / PyTorch / Ultralytics:\*\* Python 3.10.12 / PyTorch 2.5.1+cu124 / Ultralytics 8.4.145



\*\*Checkpoint:\*\* `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`



\*\*Ngưỡng detection/segmentation dùng khi tạo JSON:\*\* `0.35`



\*\*Thay đổi so với notebook nguồn:\*\* Không



#### 1\. Phân loại ảnh



\- Record hạng 1: class\_id=468 ; class\_name=cab ; rank=1 ; score0.510915 ;taxonomy\_name=ImageNet-1K

\- Phạm vi nhãn: Nói về cả bức ảnh. Vì đây là tác vụ Image Classification, kết quả không đi kèm tọa độ bounding box nào mà chỉ gán 1 nhãn duy nhất đại diện cho toàn bộ nội dung ảnh.

\- Nguồn gốc class list: Đi kèm taxonomy/checkpoint (ImageNet-1K gồm 1.000 lớp cố định), hoàn toàn không phải do mô hình tự nghĩ ra.

\- Vì sao cần ghi đủ 3 trường:

&#x20; + class\_id: Mã số định danh chuẩn, máy đọc được, tránh nhầm lẫn do lỗi chính tả hoặc ngôn ngữ.

&#x20; + class\_name: Tên lớp người đọc hiểu được.

&#x20; + taxonomy\_name: Xác định không gian nhãn (bộ từ điển nào), vì từ cab ở taxonomy ImageNet có thể khác mã số hoặc định nghĩa so với các taxonomy khác.

\- Quy định trong Guideline: Khi ảnh có nhiều vật thể (trong ảnh có cả xe buýt, ô tô con, xe tải...), guideline phải quy định rõ: Ưu tiên vật thể chiếm diện tích lớn nhất (dominant object), nằm ở trọng tâm ảnh, hoặc chuyển bài toán sang dạng đa nhãn (multi-label).

\- Không lấy score làm ground truth: Vì score chỉ là xác suất toán học/độ tự tin của mô hình, không phải chân lý. Mô hình có thể có score cao (0.51) nhưng vẫn đoán lệch (trong ảnh xe buýt màu xanh rất to và nổi bật nhưng model lại chọn cab). Ground truth bắt buộc phải do con người gán và kiểm duyệt theo quy tắc.



#### 2\. Phát hiện vật thể



Số lượng prediction theo ngưỡng:

Ngưỡng 0.20: 17 prediction (vật thể)

Ngưỡng 0.35: 11 prediction (vật thể)

Ngưỡng 0.60: 6 prediction (vật thể)



1\. Chép một record:

class\_name = person ; score = 0.912625 ; bbox\_xyxy = \[385.33, 69.24, 498.92, 348.92] ; bbox\_width = 113.58 ; bbox\_height = 279.68



2\. Giải thích box bằng lời thường:

Với gốc tọa độ (0, 0) ở góc trên cùng bên trái của ảnh:

Góc trên - trái của hộp: Nằm tại pixel (385.33, 69.24), tức cách mép trái ảnh 385.33 px và cách mép trên 69.24 px.

Góc dưới - phải của hộp: Nằm tại pixel (498.92, 348.92), tức cách mép trái ảnh 498.92 px và cách mép trên 348.92 px.

Kích thước hộp: Chiều rộng là 113.58 px (498.92−385.33498.92−385.33) và chiều cao là 279.68 px (348.92−69.24348.92−69.24). Hộp này bao trọn toàn bộ thân người của người đầu bếp đứng ở vị trí trung tâm bếp.



3\. Số lượng prediction theo 2 ngưỡng:

Ngưỡng 0.20: Có 17 prediction (xuất hiện thêm nhiều vật thể nhỏ/dễ nhầm như spoon, potted plant, bottle, dining table).

Ngưỡng 0.60: Có 6 prediction (chỉ giữ lại các vật thể có độ tin cậy cao: person, oven, bowl).



4\. Khi hạ ngưỡng, reviewer phải làm thêm việc gì:

Khi hạ ngưỡng (giữ nhiều hộp hơn), reviewer phải tốn nhiều thời gian và công sức hơn để kiểm tra, rà soát và loại bỏ các dự đoán dương tính giả (false positive - mô hình nhận diện nhầm hoặc nhiễu) cũng như xóa các hộp bị trùng lặp trên cùng một vật thể.



5\. Viết một quy tắc box cho annotator:

Bounding box phải ôm sát viền ngoài (tight bounding) của toàn bộ phần nhìn thấy được của vật thể, không để lọt khoảng trống thừa quá lớn (chỉ cho phép dung sai 2-3 pixel) và tuyệt đối không cắt lẹm vào các chi tiết của đối tượng (như tay, chân, quai cầm). Không tự ý vẽ phóng đoán phần bị che khuất hoàn toàn.



6\. Vật bị che khuất / cắt mép trên ảnh kitchen:

Vật thể quan sát được: Ở góc dưới bên trái có một cánh tay người (person 0.61) bị cắt mép khung hình và che khuất gần như toàn bộ cơ thể.

Xử lý bằng Guideline: Tình huống này nên đưa vào Guideline. Cụ thể viết thành quy tắc: "Đối với đối tượng person bị cắt mép ảnh (truncated) hoặc bị che khuất (occluded), chỉ gán nhãn nếu phần cơ thể nhìn thấy được chiếm từ 20% trở lên và có đủ đặc trưng nhận diện rõ ràng (như đầu, mặt hoặc nửa thân trên). Không gán nhãn cho các trường hợp chỉ xuất hiện một bàn tay/cánh tay lẻ loi ở mép ảnh để tránh làm nhiễu dữ liệu huấn luyện." Đưa vào guideline giúp người gán nhãn làm việc nhất quán trên toàn bộ tập dữ liệu mà không cần phải hỏi lại nhiều lần.



#### 3\. Phân đoạn theo từng đối tượng



1\. Chép một record: `instance\_id`=kitchen-001 ; `class\_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon\_point\_count`=348 ; 3–5 điểm đầu của `polygon\_xy`=\[\[446.0, 70.0], \[445.0, 71.0], \[444.0, 71.0], \[443.0, 72.0]]



2\. So với box, polygon cho thêm thông tin chi tiết về \*\*hình dạng hình học chính xác ở cấp độ từng pixel (pixel-level geometry)\*\*. Bounding box chỉ là hình chữ nhật thô bao gồm cả nhiều vùng ảnh nền, trong khi polygon bám sát từng đường viền cong, lồi, lõm của đối tượng (như vai, eo, nếp gấp tạp dề, ống quần), giúp phân biệt rõ từng pixel nào thuộc về vật thể và pixel nào là hậu cảnh.



3\. Ý nghĩa của `instance\_id`:

&#x20;  - `instance\_id` dùng để phân biệt từng cá thể/đối tượng riêng biệt xuất hiện trong cùng một bức ảnh (ví dụ `kitchen-001`, `kitchen-002`).

&#x20;  - \*\*Không phải class ID:\*\* Vì nhiều instance khác nhau vẫn có thể có cùng một class ID (ví dụ cả người đầu bếp và cánh tay ở mép bàn đều là class `person` nhưng mang hai `instance\_id` khác nhau).

&#x20;  - \*\*Không phải tracking ID:\*\* Vì nó chỉ có giá trị nhận dạng tĩnh trong bức ảnh đơn lẻ này của bài lab, không có chức năng theo dõi liên tục đối tượng qua chuỗi khung hình video theo thời gian.



4\. Quy tắc vẽ biên mask cho annotator:  

&#x20;  \*Đa giác viền (polygon mask) phải bám khít mép ngoài thực tế của vật thể, không để lấn sang pixel nền hoặc vật thể lân cận (dung sai không quá 1-2 pixel); tại các khúc uốn cong (như đầu, vai, miệng bát) phải tăng mật độ điểm đa giác để giữ đúng đường cong tự nhiên; khoảng trống hoặc khe hở giữa các bộ phận (như khoảng trống giữa hai chân người đầu bếp) phải được loại trừ (khoét lỗ), tuyệt đối không tô kín vào mask.\*



5\. Chỗ biên mờ / chạm nhau / bị che trên ảnh `kitchen`:

&#x20;  - \*\*Quan sát:\*\* Mặt bàn gỗ (`dining table 0.58`) có rất nhiều chiếc bát (`bowl`), cây lăn bột và các hũ gia vị đặt đè lên trên. Ranh giới tiếp xúc giữa đáy các chiếc bát và mặt bàn tiếp xúc trực tiếp và bị che khuất một phần.

&#x20;  - \*\*Xử lý:\*\* Sử dụng \*\*Guideline\*\*. Cụ thể viết thành quy tắc: \*"Khi có hiện tượng vật thể đè lên nhau (occlusion), các vật thể nằm phía trước (foreground như bát, cây lăn) được vẽ mask trọn vẹn phần nhìn thấy. Vật thể nằm phía sau (mặt bàn `dining table`) chỉ vẽ phần diện tích nhìn thấy được còn lại, không vẽ đè mask lên các vật thể phía trên để tránh trùng lặp pixel."



#### 4\. Vòng đời và kiểm tra chất lượng



|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|1 nhãn lớp (class\_id, class\_name) duy nhất gắn cho toàn bộ bức ảnh (Image-level label).|Ảnh traffic có nhiều phương tiện giao thông (xe buýt, ô tô con...) nhưng model chỉ gán nhãn cab (score 0.51); thiếu nhãn bao quát cảnh đường phố.|Đọc kỹ guideline để chọn nhãn của vật thể nổi bật nhất (dominant object) hoặc nhãn bao quát theo thứ tự ưu tiên đã quy định.|Kiểm tra nhãn được gán có tuân thủ đúng quy tắc ưu tiên trong guideline hay không; phát hiện lỗi gán nhãn theo cảm tính cá nhân.|
|Phát hiện vật thể|Tọa độ hộp chữ nhật bbox\_xyxy \[x\_min, y\_min, x\_max, y\_max] kèm 1 nhãn lớp cho từng vật thể riêng biệt.|Ở góc trái ảnh kitchen, chỉ xuất hiện một bàn tay thò vào mép ảnh nhưng mô hình vẫn gán nhãn person (score 0.61); các chảo treo bị che khuất đan xen.|Vẽ box ôm sát toàn bộ phần nhìn thấy của vật thể; áp dụng quy tắc diện tích tối thiểu (>= 20%) để bỏ qua các vật bị cắt mép quá mức.|Kiểm tra độ ôm sát viền của box (tightness), đảm bảo không bỏ sót vật thể và loại bỏ các hộp gán nhầm vào phần nền hoặc vật bị cắt lẹm quá nhiều.|
|Instance segmentation|Tập hợp các điểm đa giác polygon\_xy bám sát biên thực tế kèm nhãn lớp và instance\_id cho từng cá thể riêng biệt.|Chùm lá thảo mộc khô treo ở góc trên bên trái bị mô hình nhận nhầm thành potted plant (score 0.63); mặt bàn dining table bị nhiều bát đĩa đè lên.|Chấm các điểm polygon bám sát biên dạng thực của vật thể; khoét bỏ các khe hở/lỗ rỗng giữa các chi tiết; phân tách rõ từng instance độc lập.|Soi kỹ độ chính xác ở cấp độ pixel của đường biên mask (không bị lẹm, không thừa pixel nền), kiểm tra việc khoét lỗ và tách biệt giữa các instance cùng lớp.|



#### 

#### 5\. An toàn dữ liệu



Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

* Chỉ sử dụng các hình ảnh COCO công khai đã được cố định và kiểm tra mã checksum trong notebook bài lab; tuyệt đối không tải lên Colab hay GitHub bất kỳ hình ảnh cá nhân, khuôn mặt, biển số xe, dữ liệu khách hàng hoặc dữ liệu nội bộ.

Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

* Dừng ngay thao tác xử lý và thông báo ngay cho Lab Coach / Giảng viên hướng dẫn lớp học để được hỗ trợ giải quyết.



#### 6\. Checklist trước khi nộp link repo



Kiểm tra trên GitHub trong thư mục report/:



\[x] REPORT.md (file này đã điền)

\[x] day1\_lab\_outputs/classification\_predictions.json

\[x] day1\_lab\_outputs/detection\_predictions.json

\[x] day1\_lab\_outputs/segmentation\_predictions.json

\[x] day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md

\[x] day1\_lab\_outputs/visuals/classification\_top5.png

\[x] day1\_lab\_outputs/visuals/detection\_predictions.png

\[x] day1\_lab\_outputs/visuals/segmentation\_prediction.png

\[x] Ô validation cuối notebook báo PASS

\[x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output

\[x] Đã nộp link repository trên LMS




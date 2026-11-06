# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** Google Colab (CPU / GPU)

**Python / PyTorch / Ultralytics:** Python 3.12 / PyTorch 2.x / `ultralytics==8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (kiểm tra khớp SHA-256; bộ mẫu COCO: `traffic` ID `210273`, `kitchen` ID `397133`, `dining` ID `166918`)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên mặc định của notebook)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**
   - `sample_id` = `traffic` ; `class_id` = `468` ; `class_name` = `cab` ; `rank` = `1` ; `score` = `0.510915` ; `taxonomy_name` = `ImageNet-1K`

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này nói về **cả bức ảnh** (image-level label), không phải về một vật thể riêng lẻ.
   - *Lý do:* Tác vụ Image Classification của checkpoint `yolo11n-cls.pt` xuất ra một phân phối xác suất cho toàn bộ khung hình (`640 x 428` pixel) và xếp hạng top-5 lớp. Trong JSON hoàn toàn không có trường tọa độ hộp (`bbox_xyxy`) hay đa giác (`polygon_xy`) để chỉ ra vị trí vật thể nào trong ảnh là `cab`. Đáng chú ý, khi đối chiếu bằng mắt trên `classification_top5.png`, bức ảnh đường phố thực tế gồm nhiều xe buýt xanh/trắng (`minibus` đứng hạng 2 với `score = 0.164284`), ô tô con tối/bạc và người đi bộ chứ không có chiếc taxi vàng điển hình nào — cho thấy mô hình đang gán một nhãn đơn lẻ dựa trên đặc trưng tổng thể của toàn cảnh giao thông.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp **đi kèm cố định theo taxonomy của dữ liệu huấn luyện và checkpoint** (ở đây là `ImageNet-1K` gồm 1.000 lớp cố định lưu trong `yolo11n-cls.pt`).
   - Mô hình không tự nghĩ ra tên lớp mới; mạng nơ-ron chỉ tính xác suất cho 1.000 chỉ số lớp (`class_id` từ `0` đến `999`) đã được định nghĩa sẵn khi xây dựng tập dữ liệu và huấn luyện checkpoint.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - Cần ghi đủ cả 3 trường để đảm bảo tính chính xác kỹ thuật và khả năng truy nguyên (provenance):
     - `class_id`: Định danh số nguyên cố định dùng trong code huấn luyện/đánh giá, tránh lỗi do khác biệt cách viết hoa/thường, dấu gạch dưới hay bản dịch tên lớp.
     - `class_name`: Giúp con người (annotator, reviewer, kỹ sư) đọc hiểu trực quan ý nghĩa lớp mà không phải tra bảng mã.
     - `taxonomy_name`: Xác định hệ danh mục gốc (`ImageNet-1K` khác `COCO-80`). Cùng một khái niệm xe cộ nhưng ở `ImageNet-1K` tách nhỏ thành `cab`, `minibus`, `police_van`, `streetcar`, còn ở `COCO-80` gộp thành `car`, `bus`, `truck`; nếu thiếu `taxonomy_name` thì cả `class_id` lẫn `class_name` đều có thể bị hiểu sai phạm vi.

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Với bài toán phân loại đơn nhãn (single-label) khi ảnh có nhiều đối tượng cạnh tranh (như ảnh `traffic` vừa có xe buýt, ô tô con, người đi bộ, xe đạp), guideline bắt buộc phải quy định rõ:
     - *Tiêu chí chọn chủ thể chính:* Ưu tiên vật thể ở tiền cảnh, gần trung tâm hoặc chiếm diện tích lớn nhất trong khung hình.
     - *Thứ tự ưu tiên lớp (priority rule):* Nếu nhiều loại phương tiện cùng nổi bật, ưu tiên nhóm lớp nào trước (ví dụ ưu tiên phương tiện cỡ lớn như xe buýt hay lớp tổng quát).
     - *Quy tắc xử lý ảnh đa chủ thể / mơ hồ:* Quy định rõ khi nào gán nhãn "multi-object / ambiguous" hoặc escalate, hay chuyển sang bài toán multi-label / object detection thay vì để annotator chọn cảm tính.

6. **Vì sao không được lấy `score` của model làm ground truth?**
   - `score` chỉ là mức độ tự tin toán học của mô hình khi xếp hạng prediction, không phải thước đo chất lượng nhãn thật và không chứng minh dự đoán đúng. Thực tế ngay ở ảnh `traffic`, mô hình gán `cab` với `score = 0.510915` (cao gấp hơn 3 lần `minibus = 0.164284`) dù trong ảnh chủ đạo là xe buýt và ô tô thường.
   - Ground truth phải do con người tạo và kiểm tra theo guideline độc lập; nếu chép `score`/prediction của model thành ground truth sẽ biến lỗi của model thành dữ liệu "chuẩn", gây sai lệch cả tập huấn luyện lẫn kết quả đánh giá.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record (`sample_id = kitchen`):**
   - `sample_id` = `kitchen` ; `class_name` = `person` ; `score` = `0.912625` ; `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]` ; `bbox_width` = `113.58` ; `bbox_height` = `279.68`

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Trên ảnh `kitchen` kích thước `640 x 426` pixel (gốc tọa độ `(0, 0)` ở góc trên-trái bức ảnh, trục `x` tăng sang phải, trục `y` tăng xuống dưới):
     - **Góc trên-trái** của hộp nằm tại tọa độ `(x_min = 385.33, y_min = 69.24)` — tức cách mép trái ảnh `385.33` pixel và cách mép trên ảnh `69.24` pixel (ngay phía trên đầu người đầu bếp đứng bên phải).
     - **Góc dưới-phải** của hộp nằm tại tọa độ `(x_max = 498.92, y_max = 348.92)` — cách mép trái `498.92` pixel và cách mép trên `348.92` pixel (ngay dưới bàn chân người đầu bếp).
     - **Kích thước hộp:** rộng `113.58` pixel (`498.92 - 385.33`) và cao `279.68` pixel (`348.92 - 69.24`).

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - Chọn hai ngưỡng `0.35` và `0.60` trên ảnh `kitchen`:
     - Ở ngưỡng **`0.35`**: Có **11** prediction (`person` 0.91, `bowl` 0.72, `bowl` 0.70, `oven` 0.69, `oven` 0.63, `person` 0.61, `bowl` 0.50, `bowl` 0.46, `cup` 0.45, `cup` 0.38, `bowl` 0.38).
     - Ở ngưỡng **`0.60`**: Có **6** prediction (chỉ còn 6 hộp có `score >= 0.60` gồm 2 `person`, 2 `bowl`, 2 `oven`; toàn bộ 5 hộp `bowl`/`cup` có score từ `0.38` đến `0.50` bị lọc bỏ). *(Nếu hạ tiếp xuống `0.20`, số lượng prediction còn tăng cao hơn nữa do giữ thêm các hộp có model score thấp).*

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng lọc (ví dụ từ `0.60` xuống `0.35` hoặc `0.20`), số lượng hộp giữ lại tăng mạnh kéo theo nhiều **false positive** (nhận nhầm vật thể, như tủ bồn rửa bên phải bị đoán nhầm là `oven 0.63`), hộp chồng chéo hoặc hộp bao quanh chi tiết vụn.
   - Nếu dùng pre-annotation gợi ý từ model, **Reviewer / QC** phải tốn thêm công rà soát từng hộp điểm thấp để xóa hộp nhầm lớp, loại bỏ hộp trùng lặp (duplicate/overlapping boxes), chỉnh lại biên hộp lệch và vẫn phải kiểm tra các vật thể thật bị model bỏ sót (false negative). Ngưỡng chỉ lọc prediction của model, không bao giờ là quy tắc để bỏ qua vật thể thật khi tạo ground truth.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**
   - *Quy tắc:* *"Với mỗi vật thể thuộc danh sách lớp mục tiêu, hãy vẽ một hộp chữ nhật `bbox_xyxy` bám sát toàn bộ phần nhìn thấy được ngoài cùng của vật thể (tight box, sai số không quá 2 pixel ở mỗi cạnh), không cắt mất bộ phận nhìn thấy và không tự suy đoán kéo dài hộp ra ngoài vùng bị che khuất hoặc ngoài mép ảnh."*

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - *Vật thể quan sát được:* Ở sát mép trái ảnh `kitchen`, có một cánh tay/bàn tay người thò vào khung hình được model bắt thành `person` (`score = 0.610931`, `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`, bị cắt mép trái tại `x_min = 0.12`), đồng thời chiếc `cup` (`score = 0.381542`, `[119.95, 272.2, 141.78, 304.16]`) trên bàn bị che khuất một phần phía trước bởi chiếc `bowl 0.50`.
   - *Xử lý bằng Guideline hay Escalation:*
     - Đưa vào **Guideline** cho hiện tượng phổ biến lặp lại ở nhiều ảnh: quy định rõ ngưỡng cắt mép/che khuất và bộ phận tối thiểu (ví dụ: *"Nếu người hoặc vật bị cắt ở mép ảnh nhưng vẫn nhận diện rõ bộ phận đặc trưng như bàn tay/cánh tay thì vẫn gán nhãn `person` trên phần nhìn thấy; nếu vật bị che khuất > 80% hoặc quá mờ không xác định chắc chắn lớp thì bỏ qua"*).
     - Dùng **Escalation** lên Lab Coach nếu guideline hiện tại chưa quy định "chỉ thấy một bàn tay ở mép ảnh có tính là một instance `person` hay không", hoặc các vật thể tối mờ trên kệ tường không rõ là `bowl`, `cup` hay nồi chảo ngoài COCO-80 — tuyệt đối không tự đặt luật riêng.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record (`sample_id = kitchen`):**
   - `instance_id` = `kitchen-001` ; `class_name` = `person` ; `score` = `0.899318` ; số điểm polygon `polygon_point_count` = `348` ; 5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - `bbox_xyxy` chỉ cho biết 4 tọa độ khung chữ nhật bao ngoài nên gộp luôn rất nhiều pixel nền (tường gỗ, tủ bếp, sàn nhà quanh chân người đầu bếp) hoặc pixel của vật kế bên vào trong hộp.
   - `polygon_xy` mô tả **đường biên hình học sát theo từng cụm pixel** của đối tượng (như đường viền đầu, vai, tay áo, tạp dề và hai ống quần tách rời của `kitchen-001` với 348 đỉnh), nhờ đó cho biết hình dạng thật, diện tích bề mặt thật và tách biệt vùng đối tượng khỏi nền.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` (dạng `<sample_id>-NNN` như `kitchen-001`, `kitchen-002`) dùng để **phân biệt từng cá thể (instance) độc lập** trong kết quả xuất của một ảnh — ví dụ trong `kitchen` có 4 cá thể cùng lớp `bowl` (`kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010`) và 2 cá thể lớp `person` (`kitchen-001`, `kitchen-009`), mỗi cá thể có một đa giác riêng.
   - *Không phải `class_id`:* `class_id` chỉ loại vật thể trong taxonomy COCO-80 (mọi người đều có `class_id = 0`, mọi chiếc bát đều có `class_id = 45`), còn `instance_id` tách riêng từng cá thể cùng loại.
   - *Không phải `tracking_id`:* `tracking_id` dùng trong chuỗi video để giữ nguyên một mã định danh cho cùng một đối tượng xuyên qua nhiều frame liên tiếp theo thời gian, còn `instance_id` trong bài lab chỉ đánh số thứ tự cục bộ trong một ảnh tĩnh.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**
   - *Quy tắc:* *"Vẽ đa giác khép kín bám sát ranh giới thị giác thực của từng instance riêng biệt (sai số không quá 1–2 pixel), đặt thêm đỉnh ở các đoạn cong/góc khuất, không gộp hai vật thể cùng lớp chạm nhau thành một mask, và chỉ bao phủ phần diện tích thực sự nhìn thấy của vật thể (không bao gồm bóng đổ hay vật che chắn phía trước)."*

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - *Vị trí quan sát trên `segmentation_prediction.png`:* Khu vực góc dưới bên trái trên mặt bàn gỗ phủ bột mì, nơi các khuôn/bát `bowl 0.74` (`kitchen-002`), `bowl 0.53` (`kitchen-006`), `bowl 0.70` (`kitchen-003`), bàn tay `person 0.43` (`kitchen-009`) và mặt bàn `dining table 0.58` (`kitchen-005`) nằm sát/chồng lấp nhau; mặt nạ `dining table 0.58` bị tràn đè lên cả khối bột, cây cán bột và các khay bánh trên bàn. Ngoài ra góc trên-trái có bó lá khô treo bị đoán nhầm thành `potted plant 0.63` (`kitchen-004`).
   - *Dùng Guideline hay Escalation:*
     - Dùng **Guideline** cho quy tắc phân tách biên khi hai vật chạm/đặt lên nhau: quy định rõ *"Vật nằm trên mặt bàn phải được tách mask riêng và khoét/loại trừ phần diện tích bị vật phía trước che khuất khỏi mask của `dining table`; hai chiếc `bowl` chạm mép nhau phải vẽ thành hai polygon tách biệt theo gờ vật lý"*.
     - Dùng **Escalation** khi gặp trường hợp ranh giới hoặc ngữ nghĩa lớp chưa có trong guideline (ví dụ: bó thảo mộc treo ngược ở góc trái không có chậu thì có được gán `potted plant` không, hay phần bột nhào nằm tràn giữa khuôn bánh và mặt bàn tính vào đâu) để Lab Coach chốt quy ước thống nhất.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Nhãn cấp toàn ảnh (`image-level`): một mã lớp/tên lớp chuẩn (hoặc tập nhãn nếu multi-label) cho cả bức ảnh theo taxonomy dự án. | Trên `traffic`, ảnh có nhiều xe buýt xanh/trắng, ô tô con và người đi bộ nhưng model đoán `cab` hạng 1 (`score = 0.510915`), cao hơn `minibus` (`0.164284`) dù không thấy taxi rõ ràng. | Đọc kỹ guideline về chủ thể chính để chọn đúng lớp đại diện cho toàn ảnh; nếu ảnh đa chủ thể không thuộc quy tắc hiện có thì escalate. | Kiểm tra nhãn có đúng taxonomy và quy tắc chọn chủ thể chính không; bác bỏ việc chép top-1 prediction của model làm nhãn thật. |
| **Phát hiện vật thể** | Nhãn cấp vật thể (`object-level`): danh sách các hộp `bbox_xyxy` kèm `class_id`/`class_name` cho mọi vật thể hợp lệ trong ảnh. | Trên `kitchen`, tủ bồn rửa bên phải bị đoán nhầm là `oven 0.63`; mép trái chỉ lộ bàn tay bị cắt khung hình (`person 0.61`, `x_min = 0.12`); nhiều nồi/chảo/cốc nhỏ bị bỏ sót hoặc che khuất. | Gán đủ mọi vật thể mục tiêu (kể cả vật model bỏ sót), vẽ tight box bao sát phần nhìn thấy, gán đúng lớp và tuân thủ ngưỡng che khuất/cắt mép. | Kiểm tra lỗi bỏ sót (false negative), hộp dư/sai lớp (false positive), hộp lỏng hoặc cắt lẹm biên vật thể, và xử lý đồng nhất các ca cắt mép. |
| **Instance segmentation** | Nhãn cấp cá thể theo đường biên (`instance pixel/polygon-level`): mỗi cá thể có `class_id`, `instance_id` riêng và đa giác `polygon_xy` bám sát biên. | Trên `kitchen`, bó lá khô treo bị nhầm là `potted plant 0.63` (`kitchen-004`); mask `dining table 0.58` (`kitchen-005`, 598 điểm) tràn lên bột nhào và khay trên bàn; các `bowl` góc trái chạm sát nhau. | Vẽ polygon bám sát biên thật từng cá thể, tách riêng các vật cùng lớp đứng cạnh nhau (`bowl`, `spoon`), không gộp vật nằm trên bàn vào mask mặt bàn. | Phóng to kiểm tra độ sát biên polygon, lỗi tràn mask sang nền/vật khác, lỗi gộp chung nhiều cá thể thành một instance hoặc sai lớp. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**
   - Chỉ sử dụng đúng ba ảnh COCO công khai đã kiểm tra checksum trong bài lab, giữ họ tên/MSSV duy nhất ở tên repository để đối chiếu LMS và tuyệt đối không đưa ảnh cá nhân, thông tin nhận dạng (PII) hay dữ liệu nội bộ lên Colab, báo cáo hoặc GitHub công khai.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
   - Tôi sẽ dừng ngay thao tác tải lên/gán nhãn/chia sẻ dữ liệu đó và báo trực tiếp cho **Giảng viên / Lab Coach** để được hướng dẫn xử lý an toàn.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

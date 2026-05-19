# Thông tin dự án

* **📚 Môn học:** MAT3508 – Nhập môn Trí tuệ Nhân tạo 
* **📅 Học kỳ:** Học kỳ 1, Năm học 2025-2026 
* **🏫 Trường:** VNU-HUS (Đại học Quốc gia Hà Nội - Trường Đại học Khoa học Tự nhiên)  
* **📝 Tiêu đề:** Phân loại cá trong thời gian thực và tích hợp tra cứu thông tin loài cá  
* **📅 Ngày nộp:** 30/11/2025
* **📄 Báo cáo PDF:** 📄 [Liên kết tới báo cáo PDF trong kho lưu trữ này]  
* **🖥️ Slide thuyết trình:** 🖥️ [Liên kết tới slide thuyết trình trong kho lưu trữ này]  
* **📂 Kho lưu trữ:** 📁 https://drive.google.com/drive/folders/10DZXecvxZj9Ys18drgafsNuKt_pSRCJA?usp=drive_link


**👥 Thành viên nhóm:**

| 👤 Họ và tên      | 🆔 Mã sinh viên     | 🐙 Tên GitHub        | 🛠️ Đóng góp  |
|------------------|--------------------|----------------------|----------------------|
| Đặng Hải Bình    | 23001502           | chaotolabin          | Phân chia công việc<br>Tìm hiểu tổng quan và trực quan hóa bộ dữ liệu<br>Hoàn thiện, chỉnh sửa báo cáo và slide |
| Chu Thị Mai Duyên| 23001510           | maiduyen05           | Tìm hiểu, xây dựng và huấn luyện mạng nơ-ron tích chập (CNN)<br>Hoàn thiện, chỉnh sửa báo cáo và slide        |
| Đỗ Thị Mây       | 23001536           | sharonmyoui37        | Nghiên cứu, xây dựng và huấn luyện mô hình YOLO<br>Hoàn thiện, chỉnh sửa báo cáo và slide         |
| Nguyễn Trọng Đức | 23001961           | trognduck             | Ứng dụng OpenCV và xây dựng chức năng liên kết kết quả tìm kiếm<br>Hoàn thiện, chỉnh sửa báo cáo và slide            |
| Nguyễn Quốc Hiệu | 23001520           | wokhyu               | Nghiên cứu, xây dựng và huấn luyện mô hình YOLO<br>Hoàn thiện, chỉnh sửa báo cáo và slide         |

---

## 📑 Tổng quan cấu trúc báo cáo

### Chương 1: Giới thiệu

**📝 Tóm tắt dự án**  
Dự án xây dựng một **hệ thống nhận dạng và phân loại cá trong thời gian thực**, kết hợp:
- Phát hiện cá bằng **YOLOv8**
- Phân loại chi tiết loài cá bằng **CNN**
- Tích hợp **tra cứu thông tin sinh học từ Wikipedia**

Hệ thống hỗ trợ nhiều đối tượng như: nhà sinh học, trại nuôi thủy sản, ngư dân và giáo dục trực quan. Mục tiêu lâu dài là triển khai trên **thiết bị di động và hoạt động offline**.

**❓ Bài toán đặt ra**  
Việc phân loại cá thủ công:
- Tốn thời gian, chi phí nhân lực cao  
- Dễ sai sót  
- Không đáp ứng được bài toán **dữ liệu lớn** và **giám sát thời gian thực**

Dự án hướng tới tự động hóa toàn bộ quy trình **nhận dạng – phân loại – tra cứu thông tin cá** bằng thị giác máy tính.

---

### Chương 2: Phương pháp & Triển khai

**⚙️ Phương pháp**

- **Dữ liệu**: Bộ dữ liệu FishNet với **84,680 ảnh từ 463 họ cá**, ảnh thu thập từ FishBase và iNaturalist.
- **YOLOv8**:
  - Phát hiện vị trí cá trong ảnh.
  - Mô hình YOLOv8m, batch size 32, epoch 50.
- **CNN phân loại loài cá**:
  - Các kiến trúc sử dụng: ResNet-50, EfficientNet-B0, Custom CNN.
  - Batch size 64, epoch 50–120, optimizer Adam.
- **Tiền xử lý**:
  - Resize ảnh về 224×224.
  - Chuẩn hóa dữ liệu.
- **ByteTrack**:
  - Theo dõi cá giữa các frame khi chạy realtime.
- **Wikipedia API**:
  - Tra cứu thông tin sinh học, hình thái ngay sau khi nhận dạng loài.

---

**💻 Triển khai**

- **Ngôn ngữ**: Python  
- **Thư viện chính**:
  - OpenCV
  - PyTorch
  - Ultralytics YOLOv8
  - NumPy
  - Wikipedia API
  - Threading

**Pipeline thời gian thực:**
1. Nhận ảnh từ webcam
2. YOLOv8 phát hiện cá
3. ByteTrack theo dõi từng cá thể
4. Crop ảnh từng con cá
5. CNN phân loại loài
6. Ổn định nhãn (Label Stabilization)
7. Auto-Freeze khung hình
8. Tra cứu Wikipedia
9. Hiển thị kết quả realtime

**Phím điều khiển:**
- `q`: Thoát
- `SPACE`: Bật/tắt Auto-Freeze
- `L`: Mở Wikipedia loài cá

**Cấu trúc mã nguồn:**
project_root/
├── cnn/
├── docs/
├── eda/
├── utils/
├── webcam/
├── yolo/
└── README.md

### Chương 3: Kết quả & Phân tích

**📊 Kết quả & Thảo luận**

- **YOLOv8**:
  - Đạt độ chính xác cao trên tập test và validation.
  - Hoạt động tốt cả với các họ cá ít dữ liệu.
- **CNN**:
  - Độ chính xác tổng thể cao.
  - Quá trình huấn luyện hội tụ ổn định.
  - Các cặp loài dễ nhầm lẫn đã được phân tích chi tiết.
- **Phân tích dữ liệu**:
  - Dữ liệu **rất mất cân bằng** giữa các họ cá.
  - Số lượng bounding box trung bình chiếm ~46.7% diện tích ảnh.
- **Demo realtime**:
  - Hệ thống chạy ổn định với webcam.
  - Auto-Freeze giúp hiển thị thông tin trực quan.
  - Tra cứu Wikipedia không làm gián đoạn khung hình.

---

### Chương 4: Kết luận

**✅ Kết luận & Hướng phát triển**

- Hệ thống chứng minh khả năng **ứng dụng deep learning hiệu quả trong môi trường dưới nước**.
- Kết hợp thành công:
  - Nhận dạng
  - Theo dõi
  - Phân loại
  - Tra cứu tri thức
- **Hướng phát triển:**
  - Nhận dạng hành vi cá
  - Phân tích quần thể quy mô lớn
  - Triển khai trên thiết bị di động
  - Hoạt động offline

---

### Tài liệu tham khảo & Phụ lục

**📚 Tài liệu tham khảo**
- FishNet Dataset (ICCV 2023)
- YOLOv8 – Ultralytics
- ResNet, EfficientNet, MobileNet
- OpenCV, PyTorch
- Wikipedia API
- Deep Learning – Goodfellow et al.


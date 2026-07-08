# VN Car Detector

Dự án này dùng để nhận dạng biển số xe Việt Nam từ ảnh hoặc video.

Quy trình chính gồm 2 bước:

1. Dùng YOLO nano để phát hiện vùng biển số xe.
2. Cắt vùng biển số và dùng PaddleOCR để nhận diện nội dung biển số.

## Quy Trình Nhận Dạng Biển Số Xe

```txt
Khung hình ảnh/video
      |
      | Mô hình phát hiện YOLO nano
      v
Tọa độ vùng biển số
      |
      | Cắt ảnh biển số
      v
Ảnh biển số riêng
      |
      | Nhận diện ký tự bằng PaddleOCR
      v
Chuỗi ký tự biển số
```

### 1. Phát Hiện Vùng Biển Số Bằng YOLO Nano

Mô hình phát hiện biển số chính là:

```txt
data/model_trained/vietnam_car_nano_model.pt
```

Mô hình này nhận khung hình ảnh/video đầu vào và trả về khung tọa độ của vùng biển số. Mã demo đang nằm ở:

```txt
yolo/vietnam_car_detector.py
```

Với luồng chạy tối ưu OpenVINO, YOLO nano đã được xuất sang:

```txt
data/model_trained/vietnam_car_nano_model_openvino_model/vietnam_car_nano_model.xml
data/model_trained/vietnam_car_nano_model_openvino_model/vietnam_car_nano_model.bin
```

Mã chạy thời gian thực bằng OpenVINO nằm ở:

```txt
yolo/vietnam_car_opt_detector.py
```

### 2. Cắt Vùng Biển Số

Sau khi YOLO trả về tọa độ khung biển số, dự án dùng OpenCV để cắt vùng ảnh biển số:

```python
crop_img = frame[y1:y2, x1:x2]
```

Ảnh đã cắt này chỉ chứa vùng biển số, giúp OCR tập trung vào phần cần đọc và giảm nhiễu từ nền ảnh gốc.

### 3. Nhận Diện Nội Dung Biển Số Bằng PaddleOCR

OCR gốc được lấy từ mô hình nhận diện ký tự PaddleOCR:

```txt
data/latin_PP-OCRv5_mobile_rec/
```

Mô hình OCR đã được chuyển sang ONNX với kích thước đầu vào cố định:

```txt
data/model_trained/ocr_model_static.onnx
```

Sau đó mô hình ONNX được chuyển sang OpenVINO để chạy nhanh hơn:

```txt
data/model_trained/orc_vino/inference.xml
data/model_trained/orc_vino/inference.bin
data/model_trained/orc_vino/ppocrv5_latin_dict.txt
```

Trong quy trình tối ưu, đầu ra OCR được giải mã bằng CTC decoder với từ điển `ppocrv5_latin_dict.txt`, rồi trả về chuỗi ký tự biển số.

## Quy Trình OpenVINO Hiện Tại

File `yolo/vietnam_car_opt_detector.py` đang chạy đồng thời:

- YOLO nano OpenVINO để phát hiện vùng biển số.
- OCR OpenVINO chuyển từ PaddleOCR để đọc nội dung biển số.
- OpenCV để đọc video, cắt ảnh, vẽ khung biển số và hiển thị kết quả.

Đường dẫn mô hình mặc định trong file này:

```python
model_path = "../data/model_trained/vietnam_car_nano_model_openvino_model/vietnam_car_nano_model.xml"
ocr_model_path = "../data/model_trained/orc_vino/inference.xml"
```

## Vì Sao Dùng YOLO Nano Và PaddleOCR

- YOLO nano nhỏ và nhanh, phù hợp với bài toán phát hiện một class là vùng biển số.
- PaddleOCR có sẵn mô hình nhận diện ký tự mạnh, phù hợp để đọc chữ/số trên biển số sau khi đã cắt ảnh.
- Tách quy trình thành phát hiện vùng biển số và OCR giúp mỗi mô hình làm đúng nhiệm vụ của nó, dễ kiểm tra lỗi và tối ưu hơn.

## Vì Sao Chuyển Sang OpenVINO

OpenVINO giúp tối ưu suy luận trên phần cứng Intel, đặc biệt là CPU và GPU tích hợp. Khi chuyển mô hình sang định dạng OpenVINO IR (`.xml` + `.bin`), runtime có thể tối ưu graph, giảm overhead của framework và tăng tốc khi xử lý video thời gian thực.

Trong dự án này:

- YOLO nano được xuất từ PyTorch/Ultralytics sang OpenVINO.
- PaddleOCR được chuyển theo luồng `PaddleOCR -> ONNX -> OpenVINO`.
- Mã runtime chỉ cần load file `.xml`; OpenVINO sẽ tự đọc file `.bin` đi kèm.

Thông tin chi tiết về các file mô hình nằm trong:

```txt
data/model_trained/README.md
```

## Môi Trường Python

Dự án mặc định dùng Python 3.11. File `.python-version` ở root giúp các tool như `pyenv` tự động chọn đúng phiên bản.


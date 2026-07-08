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

## Quy Trình Core ML Cho macOS Và iOS

Ngoài OpenVINO cho phần cứng Intel, dự án có thêm luồng Core ML để chạy tối ưu trên hệ sinh thái Apple như MacBook Apple Silicon, iPhone và iPad. Core ML có thể tận dụng CPU, GPU hoặc Apple Neural Engine tùy thiết bị và loại mô hình.

Các file Core ML mặc định được lưu trong:

```txt
data/model_trained/coreml/
```

Sau khi export, thư mục này gồm:

```txt
data/model_trained/coreml/vietnam_car_nano_model.mlpackage
data/model_trained/coreml/latin_PP-OCRv5_mobile_rec.mlpackage
```

### Thư Viện Cần Thiết

Dự án mặc định dùng Python 3.11. Các thư viện liên quan tới export và chạy Core ML đã nằm trong `requirements.txt`:

```txt
ultralytics>=8.3,<9
coremltools>=9.0
torch
onnx>=1.16,<2
onnx2torch>=1.5,<2
onnxsim>=0.4,<0.5
opencv-python>=4.10,<5
Pillow>=10,<13
numpy>=1.26,<3
```

Cài dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Lưu ý: `torch` thường được kéo theo khi cài `ultralytics`. Nếu môi trường chưa có `torch`, cài thêm bản phù hợp với máy trước khi export Core ML.

### Chuyển YOLO Nano Sang Core ML

Mô hình YOLO nano gốc:

```txt
data/model_trained/vietnam_car_nano_model.pt
```

Script export:

```txt
yolo/export_yolo_to_coreml.py
```

Chạy export với cấu hình mặc định:

```bash
.venv/bin/python yolo/export_yolo_to_coreml.py
```

Chạy export rõ tham số:

```bash
.venv/bin/python yolo/export_yolo_to_coreml.py \
  --model data/model_trained/vietnam_car_nano_model.pt \
  --output-dir data/model_trained/coreml \
  --imgsz 640 \
  --quantize fp16
```

Output:

```txt
data/model_trained/coreml/vietnam_car_nano_model.mlpackage
```

Nếu Ultralytics báo:

```txt
'nms=True' is not available for end2end models. Forcing 'nms=False'.
```

thì đây là cảnh báo bình thường với một số model YOLO end-to-end. Khi đó Core ML model trả output raw/end-to-end, còn bước lọc confidence và xử lý box được thực hiện trong code Python hoặc trong app iOS.

### Chuyển PaddleOCR Mobile Sang Core ML

Mô hình PaddleOCR mobile gốc:

```txt
data/latin_PP-OCRv5_mobile_rec/
```

Trong project đã có sẵn ONNX static:

```txt
data/model_trained/ocr_model_static.onnx
```

Script export OCR sang Core ML:

```txt
yolo/export_paddle_ocr_to_coreml.py
```

Chạy export mặc định, dùng ONNX static đã có sẵn:

```bash
.venv/bin/python yolo/export_paddle_ocr_to_coreml.py
```

Output:

```txt
data/model_trained/coreml/latin_PP-OCRv5_mobile_rec.mlpackage
```

Luồng chuyển đổi của OCR là:

```txt
PaddleOCR inference model
      |
      | paddle2onnx, nếu cần tạo lại ONNX
      v
ONNX static, input 1x3x48x320
      |
      | onnx2torch + coremltools
      v
Core ML .mlpackage
```

Nếu muốn tạo lại ONNX từ thư mục PaddleOCR gốc trước khi export Core ML:

```bash
.venv/bin/python yolo/export_paddle_ocr_to_coreml.py --rebuild-onnx
```

Lưu ý: bước `--rebuild-onnx` cần command `paddle2onnx`. Trên macOS arm64, `paddle2onnx` có thể xung đột dependency với các bản ONNX mới, nên cách ổn định hơn là dùng file `data/model_trained/ocr_model_static.onnx` đã có sẵn.

OCR Core ML vẫn trả về logits theo chuỗi thời gian. Để lấy ra chuỗi biển số, runtime cần dùng CTC decoder với dictionary:

```txt
data/model_trained/orc_vino/ppocrv5_latin_dict.txt
```

### Chạy Detect Bằng Core ML

File chạy suy luận Core ML:

```txt
yolo/vietnam_car_opt_coreml.py
```

Script này load đồng thời:

- `vietnam_car_nano_model.mlpackage` để phát hiện vùng biển số.
- `latin_PP-OCRv5_mobile_rec.mlpackage` để nhận diện nội dung biển số.
- `ppocrv5_latin_dict.txt` để decode output OCR bằng CTC.

Chạy với video mặc định `data/test.MOV`:

```bash
.venv/bin/python yolo/vietnam_car_opt_coreml.py
```

Chạy với video khác:

```bash
.venv/bin/python yolo/vietnam_car_opt_coreml.py \
  --video data/test.MOV \
  --detector-model data/model_trained/coreml/vietnam_car_nano_model.mlpackage \
  --ocr-model data/model_trained/coreml/latin_PP-OCRv5_mobile_rec.mlpackage
```

Giới hạn tốc độ vòng lặp xử lý/hiển thị tối đa 60 FPS:

```bash
.venv/bin/python yolo/vietnam_car_opt_coreml.py --max-fps 60
```

Tắt giới hạn FPS:

```bash
.venv/bin/python yolo/vietnam_car_opt_coreml.py --max-fps 0
```

Trong app iOS, có thể đưa hai file `.mlpackage` vào Xcode, dùng CoreML/Vision để chạy YOLO, crop vùng biển số, resize/pad ảnh OCR về `1x3x48x320`, sau đó decode logits bằng CTC với dictionary tương ứng.

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

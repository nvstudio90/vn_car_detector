# Model Trained

Thư mục này chứa các mô hình đã train/xuất cho quy trình nhận diện biển số xe Việt Nam:

1. Phát hiện vùng biển số trong khung hình ảnh/video.
2. Cắt vùng biển số.
3. OCR ký tự trên biển số.

Quy trình tối ưu hiện tại dùng OpenVINO cho suy luận. Có thể xem ví dụ trong `yolo/vietnam_car_opt_detector.py`.

## Các File Mô Hình

| Đường dẫn | Dùng để làm gì |
| --- | --- |
| `car_best.pt` | Mô hình YOLO/PyTorch phát hiện biển số, có thể là bản train cũ hoặc bản thử nghiệm. |
| `vietnam_car_model.pt` | Mô hình YOLO/PyTorch phát hiện biển số, kích thước đầy đủ hơn bản nano. |
| `vietnam_car_nano_model.pt` | Mô hình YOLO/PyTorch phát hiện biển số bản nhỏ, dùng làm nguồn để xuất sang OpenVINO. |
| `quick_draw_model.pt` | Mô hình PyTorch riêng cho chức năng quickdraw/demo. File này không nằm trong quy trình OpenVINO nhận diện biển số chính. |
| `ocr_model_static.onnx` | Mô hình OCR đã chuyển sang ONNX với kích thước đầu vào cố định. Đây là file trung gian trước khi chuyển sang OpenVINO. |

## Các Thư Mục OpenVINO

### `vietnam_car_nano_model_openvino_model/`

Đây là bản OpenVINO IR của `vietnam_car_nano_model.pt`.

| File | Ý nghĩa |
| --- | --- |
| `vietnam_car_nano_model.xml` | Mô tả graph/cấu trúc mô hình của OpenVINO IR. |
| `vietnam_car_nano_model.bin` | Trọng số của mô hình. File `.xml` cần file `.bin` đi kèm để chạy suy luận. |
| `metadata.yaml` | Metadata do Ultralytics xuất ra, gồm task, input size, class name và cấu hình xuất mô hình. |

Metadata hiện tại:

- Task: `detect`
- Input size: `640x640`
- Class: `VietNamCarLicenseTest`
- Export: `half: true`, `int8: false`

### `vietnam_car_nano_model_int8_openvino_model/`

Đây là bản OpenVINO IR đã lượng tử hóa INT8 của mô hình nano.

| File | Ý nghĩa |
| --- | --- |
| `vietnam_car_nano_model.xml` | Graph/cấu trúc mô hình của bản INT8. |
| `vietnam_car_nano_model.bin` | Trọng số đã lượng tử hóa. |
| `metadata.yaml` | Metadata của lần xuất mô hình. |

Metadata hiện tại:

- Task: `detect`
- Input size: `320x320`
- Class: `VietNamCarLicenseTest`
- Export: `half: false`, `int8: true`

Bản INT8 thường nhỏ hơn và có thể nhanh hơn trên CPU Intel, nhưng cần kiểm tra lại độ chính xác vì lượng tử hóa có thể làm giảm chất lượng nhận diện.

### `orc_vino/`

Thư mục này chứa mô hình OCR sau khi chuyển từ `ocr_model_static.onnx` sang OpenVINO IR.

| File | Ý nghĩa |
| --- | --- |
| `inference.xml` | Graph/cấu trúc mô hình của OCR OpenVINO. |
| `inference.bin` | Trọng số của mô hình OCR. |
| `ppocrv5_latin_dict.txt` | Từ điển ký tự cho CTC decoder khi giải mã đầu ra OCR. |

Lưu ý: tên thư mục hiện tại là `orc_vino`. Về mặt ý nghĩa, đây là mô hình OCR VINO. Nếu đổi tên thành `ocr_vino` thì cần cập nhật các đường dẫn trong code.

## Vì Sao Chuyển Sang OpenVINO

OpenVINO là runtime tối ưu suy luận của Intel. Khi chuyển mô hình từ PyTorch/ONNX sang OpenVINO IR (`.xml` + `.bin`), runtime có thể tối ưu graph và kernel theo phần cứng Intel tốt hơn so với chạy trực tiếp bằng framework gốc.

Lợi ích chính:

- Tăng tốc suy luận trên CPU Intel, GPU tích hợp Intel và các thiết bị Intel hỗ trợ OpenVINO.
- Giảm overhead của Python/framework khi chạy video thời gian thực.
- Hỗ trợ tối ưu FP16/INT8 để giảm kích thước mô hình và tăng throughput.
- Có thể compile mô hình một lần bằng `openvino.Core().compile_model(...)`, sau đó gọi suy luận lặp lại nhanh hơn trong vòng lặp video.
- Định dạng IR gồm `.xml` và `.bin` ổn định hơn cho triển khai so với việc phụ thuộc trực tiếp vào PyTorch/Paddle trong runtime.

Trong dự án này:

- YOLO detector được xuất bằng `YOLO(...).export(format="openvino", ...)`.
- OCR được chuyển theo luồng `PaddleOCR -> ONNX -> OpenVINO`.
- Code tối ưu đọc mô hình bằng `core.read_model(...)`, compile bằng `core.compile_model(...)`, rồi suy luận trực tiếp trong `yolo/vietnam_car_opt_detector.py`.

## Nguồn Gốc Và Luồng Chuyển Đổi OCR

File `ocr_model_static.onnx` được tạo từ mô hình nhận diện ký tự PaddleOCR trong thư mục:

```txt
data/latin_PP-OCRv5_mobile_rec/
```

Thư mục PaddleOCR gốc có các file quan trọng:

| File | Ý nghĩa |
| --- | --- |
| `inference.json` | Mô tả cấu trúc Paddle inference model. |
| `inference.pdiparams` | Trọng số của mô hình Paddle. |
| `inference.yml` | Cấu hình tiền xử lý/hậu xử lý, kích thước đầu vào và từ điển. |
| `config.json` | Metadata/cấu hình mô hình, bao gồm `model_name: latin_PP-OCRv5_mobile_rec`. |

Luồng chuyển đổi OCR đầy đủ:

```txt
data/latin_PP-OCRv5_mobile_rec/inference.json
data/latin_PP-OCRv5_mobile_rec/inference.pdiparams
        |
        | paddle2onnx
        v
model_dynamic.onnx
        |
        | onnxsim --overwrite-input-shape x:1,3,48,320
        v
ocr_model_static.onnx
        |
        | openvino.convert_model(...)
        v
data/model_trained/orc_vino/inference.xml
data/model_trained/orc_vino/inference.bin
```

Trong `yolo/yolo_util.py` đang lưu lại lệnh đã dùng trên Google Colab:

```bash
pip install paddle2onnx onnxsim
pip install paddlepaddle

paddle2onnx --model_dir . \
             --model_filename inference.json \
             --params_filename inference.pdiparams \
             --save_file model_dynamic.onnx \
             --opset_version 11

onnxsim model_dynamic.onnx model_static.onnx --overwrite-input-shape x:1,3,48,320
```

Sau bước này, `model_static.onnx` được đưa vào dự án với tên `ocr_model_static.onnx`.

Có thể kiểm tra dấu vết chuyển đổi bằng cách đọc string trong ONNX:

```bash
strings data/model_trained/ocr_model_static.onnx | rg "p2o.pd_op|PaddlePaddle"
```

Nếu thấy các node dạng `p2o.pd_op...` và graph name `PaddlePaddle Graph in PIR mode` thì đây là dấu vết mô hình được chuyển từ Paddle sang ONNX bằng Paddle2ONNX.

File OpenVINO `data/model_trained/orc_vino/inference.xml` cũng có metadata:

```xml
<input_model value="DIR/ocr_model_static.onnx" />
```

Dòng này xác nhận mô hình OCR OpenVINO hiện tại được chuyển từ `ocr_model_static.onnx`.

## Cách Tạo Lại Mô Hình OpenVINO

Xuất YOLO nano sang OpenVINO:

```python
from ultralytics import YOLO

model = YOLO("../data/model_trained/vietnam_car_nano_model.pt")
model.export(format="openvino", imgsz=640, half=True)
```

Chuyển OCR ONNX sang OpenVINO:

```python
import os
import openvino as ov

model_onnx_path = "../data/model_trained/ocr_model_static.onnx"
save_dir = "../data/model_trained/orc_vino"

ov_model = ov.convert_model(model_onnx_path)
os.makedirs(save_dir, exist_ok=True)
ov.save_model(ov_model, os.path.join(save_dir, "inference.xml"), compress_to_fp16=False)
```

## Cách Chạy Quy Trình OpenVINO

Trong `yolo/vietnam_car_opt_detector.py`, quy trình hiện đang dùng:

```python
model_path = "../data/model_trained/vietnam_car_nano_model_openvino_model/vietnam_car_nano_model.xml"
ocr_model_path = "../data/model_trained/orc_vino/inference.xml"
```

Khi chạy OpenVINO, chỉ cần truyền đường dẫn file `.xml`; OpenVINO sẽ tự đọc file `.bin` cùng thư mục.

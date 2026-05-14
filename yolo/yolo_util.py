import os
import openvino as ov
from paddleocr import PaddleOCR
from ultralytics import YOLO

# chuyển đổi paddle model sang onnx trên google colab
# !pip install paddle2onnx onnxsim
# !pip install paddlepaddle
#
# import paddle2onnx
#
# !paddle2onnx --model_dir . \
#              --model_filename inference.json \
#              --params_filename inference.pdiparams \
#              --save_file model_dynamic.onnx \
#              --opset_version 11
#
# !onnxsim model_dynamic.onnx model_static.onnx --overwrite-input-shape x:1,3,48,320

def download_paddle_model():
    ocr = PaddleOCR(use_textline_orientation=True,
                    lang='vi',
                    text_detection_model_dir=None,  # Tự động tải mô hình mobile mặc định
                    text_recognition_model_dir=None)

# chuyển đổi yolo model sang openvino để tăng tốc trên máy chip intel
def export_to_vino(model_path):
    yolo = YOLO(model_path)
    yolo.export(format = "openvino", imgsz=640, half=True)


# chuyển đổi paddle model sang open vino thông qua onnx
def export_via_onnx(model_onnx_path, save_dir):
    try:
        print("\n--- Chuyển ONNX sang OpenVINO ---")
        ov_model = ov.convert_model(model_onnx_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ov.save_model(ov_model, os.path.join(save_dir, "inference.xml"), compress_to_fp16=False)
        print(f"Hoàn tất! File OpenVINO sẵn sàng tại: {save_dir}")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == '__main__':
    # export_to_vino("../data/model_trained/vietnam_car_nano_model.pt")
    export_via_onnx("../data/model_trained/ocr_model_static.onnx", save_dir="../data/model_trained/orc_vino")

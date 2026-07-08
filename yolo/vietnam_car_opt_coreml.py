import argparse
import time
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COREML_TMP_DIR = PROJECT_ROOT / ".tmp/coreml"
COREML_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(COREML_TMP_DIR))

import coremltools as ct

from yolo.ctc_decoder import PureCTCDecoder

DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data/test.MOV"
DEFAULT_DETECTOR_MODEL_PATH = (
    PROJECT_ROOT / "data/model_trained/coreml/vietnam_car_nano_model.mlpackage"
)
DEFAULT_OCR_MODEL_PATH = (
    PROJECT_ROOT / "data/model_trained/coreml/latin_PP-OCRv5_mobile_rec.mlpackage"
)
DEFAULT_DICT_PATH = PROJECT_ROOT / "data/model_trained/orc_vino/ppocrv5_latin_dict.txt"

CONF_THRESHOLD = 0.8
OCR_CONF_THRESHOLD = 0.8
DEFAULT_MAX_FPS = 60.0

decoder = PureCTCDecoder(str(DEFAULT_DICT_PATH))


def _first_model_input(model):
    return model.get_spec().description.input[0]


def _first_model_output_name(model):
    return model.get_spec().description.output[0].name


def _multi_array_shape(feature):
    return tuple(int(dim) for dim in feature.type.multiArrayType.shape)


def _image_size(feature):
    return int(feature.type.imageType.width), int(feature.type.imageType.height)


def preprocess_detector_image(frame, target_size=640):
    h, w = frame.shape[:2]
    r = min(target_size / h, target_size / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw = target_size - new_unpad_w
    dh = target_size - new_unpad_h
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    return image, (left, top, r)


def calculate_ratio(width, height, target_w=320, target_h=48):
    return min(target_w / width, target_h / height)


def pre_ocr_image(frame, target_width, target_height):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    r = calculate_ratio(w, h, target_width, target_height)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = target_width - new_unpad_w, target_height - new_unpad_h
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_norm = (img_float - mean) / std

    img_tensor = img_norm.transpose((2, 0, 1))
    return np.expand_dims(img_tensor, axis=0).astype(np.float32)


def postprocess_detector_output(result, letterbox_info, frame_shape):
    left, top, r = letterbox_info
    frame_h, frame_w = frame_shape[:2]
    output = np.squeeze(result)
    final_results = []

    if output.ndim == 1:
        output = np.expand_dims(output, axis=0)

    for row in output:
        if len(row) < 6:
            continue

        confidence = float(row[4])
        if confidence < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = row[:4]
        class_id = int(row[5])

        xx1 = int(max(0, min(frame_w - 1, round((x1 - left) / r))))
        yy1 = int(max(0, min(frame_h - 1, round((y1 - top) / r))))
        xx2 = int(max(0, min(frame_w - 1, round((x2 - left) / r))))
        yy2 = int(max(0, min(frame_h - 1, round((y2 - top) / r))))

        if xx2 <= xx1 or yy2 <= yy1:
            continue

        final_results.append(
            {
                "confidence": confidence,
                "class": class_id,
                "box": [xx1, yy1, xx2 - xx1, yy2 - yy1],
            }
        )

    return final_results


def run_coreml_model(model, input_name, output_name, value):
    result = model.predict({input_name: value})
    return result[output_name]


def extract_bienso_coreml(ocr_frame, ocr_model, input_name, output_name):
    result_coreml = run_coreml_model(ocr_model, input_name, output_name, ocr_frame)
    text, conf = decoder.decode(result_coreml)
    if text and conf >= OCR_CONF_THRESHOLD:
        print(f"Đã tìm thấy biển số {text}, độ tin cậy {conf:.4f}")
        return text, conf
    return "", 0.0


def car_video_detector_coreml(
    video_path=DEFAULT_VIDEO_PATH,
    detector_model_path=DEFAULT_DETECTOR_MODEL_PATH,
    ocr_model_path=DEFAULT_OCR_MODEL_PATH,
    conf_threshold=CONF_THRESHOLD,
    ocr_conf_threshold=OCR_CONF_THRESHOLD,
    max_fps=DEFAULT_MAX_FPS,
):
    global CONF_THRESHOLD, OCR_CONF_THRESHOLD
    CONF_THRESHOLD = conf_threshold
    OCR_CONF_THRESHOLD = ocr_conf_threshold

    detector_model_path = Path(detector_model_path).resolve()
    ocr_model_path = Path(ocr_model_path).resolve()
    video_path = Path(video_path).resolve()

    detector_model = ct.models.MLModel(str(detector_model_path), compute_units=ct.ComputeUnit.ALL)
    ocr_model = ct.models.MLModel(str(ocr_model_path), compute_units=ct.ComputeUnit.ALL)

    detector_input = _first_model_input(detector_model)
    detector_input_name = detector_input.name
    detector_output_name = _first_model_output_name(detector_model)
    detector_input_w, detector_input_h = _image_size(detector_input)

    ocr_input = _first_model_input(ocr_model)
    ocr_input_name = ocr_input.name
    ocr_output_name = _first_model_output_name(ocr_model)
    _, _, ocr_input_h, ocr_input_w = _multi_array_shape(ocr_input)

    print(f"Kích thước đầu vào detector Core ML: {detector_input_w}x{detector_input_h}")
    print(f"Kích thước đầu vào OCR Core ML: {ocr_input_w}x{ocr_input_h}")
    print(f"Detector output: {detector_output_name}")
    print(f"OCR output: {ocr_output_name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    min_frame_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0

    while cap.isOpened():
        frame_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break

        detector_image, info = preprocess_detector_image(frame, target_size=detector_input_w)
        detector_output = run_coreml_model(
            detector_model,
            detector_input_name,
            detector_output_name,
            detector_image,
        )
        results = postprocess_detector_output(detector_output, info, frame.shape)

        for ret in results:
            x1, y1, box_w, box_h = ret["box"]
            x2 = x1 + box_w
            y2 = y1 + box_h

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop_img = frame[y1:y2, x1:x2]

            ocr_frame = pre_ocr_image(crop_img, ocr_input_w, ocr_input_h)
            if ocr_frame is None:
                continue

            text, _ = extract_bienso_coreml(
                ocr_frame,
                ocr_model,
                ocr_input_name,
                ocr_output_name,
            )
            if text:
                cv2.putText(
                    frame,
                    text,
                    (x1, max(30, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    4,
                )

        elapsed = time.perf_counter() - frame_start
        if min_frame_interval > 0 and elapsed < min_frame_interval:
            time.sleep(min_frame_interval - elapsed)
            elapsed = time.perf_counter() - frame_start

        fps = 1 / max(elapsed, 1e-6)
        cv2.putText(
            frame,
            f"FPS {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Optimized YOLO Core ML", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Chạy nhận dạng biển số bằng YOLO Core ML và PaddleOCR Core ML."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Đường dẫn video đầu vào. Mặc định: {DEFAULT_VIDEO_PATH}",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL_PATH,
        help=f"YOLO Core ML .mlpackage. Mặc định: {DEFAULT_DETECTOR_MODEL_PATH}",
    )
    parser.add_argument(
        "--ocr-model",
        type=Path,
        default=DEFAULT_OCR_MODEL_PATH,
        help=f"OCR Core ML .mlpackage. Mặc định: {DEFAULT_OCR_MODEL_PATH}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="Ngưỡng confidence cho detector. Mặc định: 0.8.",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=OCR_CONF_THRESHOLD,
        help="Ngưỡng confidence cho OCR CTC decoder. Mặc định: 0.8.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=DEFAULT_MAX_FPS,
        help="Giới hạn FPS tối đa cho vòng lặp xử lý/hiển thị. Đặt 0 để tắt giới hạn.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    car_video_detector_coreml(
        video_path=args.video,
        detector_model_path=args.detector_model,
        ocr_model_path=args.ocr_model,
        conf_threshold=args.conf,
        ocr_conf_threshold=args.ocr_conf,
        max_fps=args.max_fps,
    )

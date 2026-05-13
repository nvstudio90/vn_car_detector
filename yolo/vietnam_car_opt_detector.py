import time

import cv2
import numpy as np
from ultralytics import YOLO
import openvino

CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.50

def car_video_detector(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success, frame = cap.read()
        results = model.predict(frame)
        if success:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 1. Lấy tọa độ khung (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Chuyển về số nguyên

                    # 2. Lấy độ tin cậy (Confidence)
                    conf = round(float(box.conf[0]), 2)

                    # 3. Lấy ID lớp và tên lớp
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # 4. Vẽ lên ảnh bằng OpenCV
                    # Vẽ hình chữ nhật
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # In thông tin ra console nếu cần
                    print(f"Detected: {label} tại [{x1}, {y1}, {x2}, {y2}] với độ tự tin {conf}")

            cv2.imshow("YOLO Video Detection", frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def preprocess(frame, target_size = 640):
    h, w = frame.shape[:2]
    r = min(target_size / h, target_size / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    dw = target_size - new_unpad_w  # Tổng chiều ngang còn thiếu
    dh = target_size - new_unpad_h  # Tổng chiều dọc còn thiếu
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose((2, 0, 1))  # HWC -> CHW
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0
    return img_tensor, (left, top, r)


def preprocess(frame, target_size=640):
    h, w = frame.shape[:2]
    r = min(target_size / h, target_size / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))

    img = cv2.resize(frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = target_size - new_unpad_w, target_size - new_unpad_h
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # Chuẩn hóa và chuyển sang NCHW
    img_tensor = img_rgb.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0
    return img_tensor, (left, top, r)


def postprocess(result, letterbox_info):
    left, top, r = letterbox_info
    output = np.squeeze(result)
    final_results = []
    for row in output:
        confidence = float(row[4])
        if confidence > 0.5:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            class_id = int(row[5])
            xx1 = (x1 - left) / r
            yy1 = (y1 - top) / r
            xx2 = (x2 - left) / r
            yy2 = (y2 - top) / r
            final_results.append({
                "confidence": confidence,
                "class": class_id,
                "box": [int(xx1), int(yy1), int(xx2 - xx1), int(yy2 - yy1)],
            })
    return final_results


def car_video_detector2(video_path, model_path):
    core = openvino.Core()
    model_onx = core.read_model(model_path)
    compiled_model = core.compile_model(model_onx)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Lấy kích thước ảnh đầu vào mà model yêu cầu (thường là 1, 3, 640, 640)
    _, _, input_h, input_w = input_layer.shape
    print(f"Kích thước đầu vào của mô hình: {input_w}x{input_h}")

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        t1 = time.time()
        # Tiền xử lý
        transformed_frame, info = preprocess(frame, target_size=640)
        left, top, r = info
        # Inference
        results = compiled_model([transformed_frame])[output_layer]

        # Hậu xử lý (Nay đã cực nhanh nhờ Numpy)
        results = postprocess(results, letterbox_info=info)
        t2 = time.time()
        print("dt = ", (t2 - t1) * 1000)
        fps = 1 / (t2 - t1)
        for ret in results:
           box = ret["box"]
           x1 = box[0]
           y1 = box[1]
           x2 = box[2] + x1
           y2 = box[3] + y1
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"FPS {fps}"
        cv2.putText(frame, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Optimized YOLO OpenVINO", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "../data/test.MOV"
    model_path = "../data/model_trained/vietnam_car_nano_model_openvino_model/vietnam_car_nano_model.xml"
    car_video_detector2(path, model_path)
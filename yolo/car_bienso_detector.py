from ultralytics import YOLO
import cv2

def test(image_path):
    model = YOLO("../data/model_trained/car_best.pt")
    results = model.predict(image_path, imgsz = 640)
    img = cv2.imread(image_path)
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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Ghi tên nhãn và độ tin cậy
            text = f"{label} {conf}"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # In thông tin ra console nếu cần
            print(f"Detected: {label} tại [{x1}, {y1}, {x2}, {y2}] với độ tự tin {conf}")

    # Show kết quả cuối cùng
    cv2.imshow("Custom Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = "/Users/om/Downloads/CarLicensePlate/images/Cars426.png"
    test(path)
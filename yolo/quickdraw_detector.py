import cv2
from ultralytics import YOLO
import numpy as np

from yolo.yolo_util import pre_ocr_image


def quickdraw_detector(model_path, image):
    yolo = YOLO(model_path)
    results = yolo.predict(image, conf=0.2)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])
            name = yolo.names[cls_id]
            print(r"class: ", name, "conf:", conf)


if __name__ == '__main__':
    image_data = cv2.imread('../data/cat_fp_test2.jpg')
    # image_data = pre_ocr_image(image_data, 640, 640)
    # kernel = np.ones((3, 3), np.uint8)
    # img_thick = cv2.erode(image_data, kernel, iterations=1)
    # img_color = cv2.cvtColor(img_thick, cv2.COLOR_GRAY2BGR)
    quickdraw_detector("../data/model_trained/quick_draw_model.pt", image_data)
    # cv2.imshow("quick_draw_model", image_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


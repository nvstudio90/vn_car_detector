import cv2
import numpy as np
from ultralytics import YOLO

from yolo.yolo_util import preprocess_image

def crop_object(image_data, x1, y1, x2, y2, letterbox_info):
    left, top, r = letterbox_info
    x1 = (x1 - left) / r
    y1 = (y1 - top) / r
    x2 = (x2 - left) / r
    y2 = (y2 - top) / r
    return image_data[int(y1):int(y2), int(x1):int(x2)]

def remove_background_and_smooth(object_frame):
    h, w = object_frame.shape[:2]
    mask = np.ones((h, w), np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin = 150

    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    # 5. Chạy thuật toán GrabCut
    # iterCount=5: Số lần lặp để tối ưu hóa nét cắt (thường từ 5-7 là đủ tốt)
    # mode=cv2.GC_INIT_WITH_RECT: Khởi tạo thuật toán dựa trên khung hình chữ nhật phía trên
    cv2.grabCut(object_frame, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    # 6. Hậu xử lý mặt nạ đầu ra
    # Sau khi chạy, mask sẽ được cập nhật chứa 4 giá trị từ 0 đến 3:
    # 0: Chắc chắn là nền (Definite Background)
    # 1: Chắc chắn là vật thể (Definite Foreground)
    # 2: Có thể là nền (Probable Background)
    # 3: Có thể là vật thể (Probable Foreground)
    #
    # Ta chỉ muốn giữ lại những vùng là vật thể (giá trị 1 hoặc 3)
    # mask_binary = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
    # # Làm mờ mượt mà đường biên bằng GaussianBlur
    # mask_smooth = cv2.GaussianBlur(mask_eroded, (5, 5), 0)
    # b, g, r = cv2.split(object_frame)
    # rgba_result = cv2.merge([b, g, r, mask_smooth])
    # return rgba_result

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def create_mask_and_filter(original_img, contour):
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    white_background = np.ones(original_img.shape, dtype=np.uint8) * 255
    mask_smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    result_white = np.where(mask_smoothed[:, :, np.newaxis] == 255, original_img, white_background)
    return result_white

def create_mask_and_filter2(original_img, contour):
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_32f = gray.astype(np.float32) / 255.0
    mask_32f = mask.astype(np.float32) / 255.0
    refined_mask = cv2.ximgproc.guidedFilter(
        guide=gray_32f,
        src=mask_32f,
        radius=8,
        eps=1e-4
    )
    refined_mask = (refined_mask * 255.0).astype(np.uint8)
    _, final_mask = cv2.threshold(refined_mask, 128, 255, cv2.THRESH_BINARY)
    # white_background = np.ones(original_img.shape, dtype=np.uint8) * 255
    # result_white = np.where(final_mask[:, :, np.newaxis] == 255, original_img, white_background)
    # return result_white
    return cv2.bitwise_and(original_img, original_img, mask = final_mask)
    # return (refined_mask * 255.0).astype(np.uint8)


def extract_object(model, image_data):
    normalized_image, letterbox_info = preprocess_image(image_data, 640, 640)
    results = model.predict(normalized_image, conf = 0.75)
    left, top, r = letterbox_info
    pad = np.array([left, top], dtype=np.float32)  # Thay bằng biến left, top của bạn
    ratio = np.array([r, r], dtype=np.float32)
    for r in results:
        if r.masks is not None:
            for i, seg in enumerate(r.masks.xy):
                # Chuyển đổi tọa độ sang kiểu số nguyên (int32) để OpenCV hiểu được
                seg_transformed = seg.copy()
                seg_transformed  = (seg_transformed - pad) / ratio
                contour = seg_transformed.astype(np.int32).reshape(-1, 1, 2)

                # Lấy tên của vật thể (ví dụ: 'person', 'cup', 'cell phone'...)
                class_id = int(r.boxes.cls[i])
                label = model.names[class_id]
                result_white = create_mask_and_filter2(image_data, contour)
                # isolated_object = cv2.bitwise_and(image_data, image_data, mask=mask_smoothed)
                cv2.imshow('mask', result_white)
                break

                # 4. Vẽ ĐƯỜNG BIÊN lên ảnh gốc bằng OpenCV
                # (0, 255, 0) là màu xanh lá, số 2 là độ dày đường viền
                # cv2.drawContours(image_data, [contour], -1, (0, 255, 0), 2)

                # 5. MẸO: Nếu bạn muốn TÁCH NỀN (chỉ giữ lại vật thể)
                # Tạo một mặt nạ đen, vẽ contour màu trắng đè lên và cắt ảnh
                # mask_black = np.zeros(frame.shape[:2], dtype=np.uint8)
                # cv2.drawContours(mask_black, [contour], -1, 255, cv2.FILLED)
                # isolated_object = cv2.bitwise_and(frame, frame, mask=mask_black)

        # object_frame = remove_background_and_smooth(r.object_frame)
        # for box in r.boxes:
        #     # conf = round(float(box[0]), 2)
        #     x1, y1, x2, y2 = box.xyxy[0]
        #     x1 = int(x1)
        #     y1 = int(y1)
        #     x2 = int(x2)
        #     y2 = int(y2)
        #     obj = crop_object(image_data, x1, y1, x2, y2, letterbox_info)
        #     result = remove_background_and_smooth(obj)
        #     cv2.imshow("object", result)
        #     break


    cv2.imshow("YOLO AI Edge Detection", image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = YOLO("yolo26m-seg.pt")
    image = cv2.imread("../data/cat_test.webp")
    extract_object(model, image)


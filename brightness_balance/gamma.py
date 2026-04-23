import cv2
import numpy as np
# from functools import partial
# import sys
# from pathlib import Path
# root_path = str(Path(__file__).resolve().parent.parent)
# if root_path not in sys.path:
#     sys.path.append(root_path)

from util.image_util import resize_image

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def gamma_callback(image, window_name):
    def callback(val):
        gamma_val = val / 10.0
        if gamma_val == 0: gamma_val = 0.1  # Đảm bảo không bằng 0

        # Thực hiện hiệu chỉnh
        result = adjust_gamma(image, gamma_val)

        # Ghi giá trị Gamma hiện tại lên ảnh để dễ quan sát
        cv2.putText(result, f"Gamma: {gamma_val}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, result)
    return callback


def test(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Không tìm thấy file ảnh!")
        exit()
    original_img = resize_image(original_img, target_width=1080, target_height=None)
    window_name = "Gamma Correction Test"
    callback = gamma_callback(original_img, window_name)
    # cv2.imshow("Original Image", original_img)
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Gamma x10", "Gamma Correction Test", 10, 50, callback)
    callback(10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
   test("/Users/om/Desktop/data/low-light.jpg")

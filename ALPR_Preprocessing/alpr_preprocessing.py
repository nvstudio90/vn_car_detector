import cv2

def alpr_preprocessing(image: cv2.Mat, limit = 2.0):
    # chuyển sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)
    # dùng CLAHE để cân bằng sáng
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    vc = clahe.apply(v)
    hsv_image = cv2.merge((h,s,vc))
    color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # làm mờ ảnh để xoá các chi tiết nhiễu
    blurring_image = cv2.GaussianBlur(grey_image, (9,9), 0)
    edges = cv2.Canny(blurring_image, 50, 150)
    return edges

def apply_alpr_and_show(image: cv2.Mat, window_name: str):
    def callback(limit):
        print("input limit: ", limit)
        ret = alpr_preprocessing(image, limit / 10.0)
        cv2.imshow(window_name, ret)
    return callback

def handle_preprocessing(image_path: str):
    image = cv2.imread(image_path)
    window_name = "Preprocessing image"
    callback = apply_alpr_and_show(image, window_name)
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Preprocessing x10", window_name, 10, 50, callback)
    callback(10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = "/Users/om/Desktop/data/bienso.png"
    handle_preprocessing(path)


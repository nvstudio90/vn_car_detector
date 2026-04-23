import cv2
from util.image_util import resize_image


def apply_clahe_gray_image(image, limit = 3.0):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    return clahe.apply(grey_image)

def apply_clahe_color_image(image, limit = 2.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    l_img = cv2.merge((cl,a,b))
    return cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)

def image_clahe_apply_callback(image, window_name):
    def callback(limit):
        ret = apply_clahe_color_image(image, limit / 10.0)
        cv2.imshow(window_name, ret)
    return callback

def test_color_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        return
    img = resize_image(img, target_width=1080)
    window_name = "CLAHE_image"
    cv2.namedWindow(window_name)
    callback = image_clahe_apply_callback(img, window_name)
    cv2.createTrackbar("CLAHE Limit x10", window_name, 10, 50, callback)
    callback(10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_color_image("/Users/om/Desktop/data/low-light.jpg")




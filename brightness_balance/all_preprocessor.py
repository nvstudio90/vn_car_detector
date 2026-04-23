import cv2
from brightness_balance.gamma import adjust_gamma
from util.image_util import resize_image

# kết hợp giữa phương pháp GAMMA và CLAHE
def master_preprocessor(image):
    gamma_img = adjust_gamma(image, 1.2)
    lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    l_img = cv2.merge((cl,a,b))
    return cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)

def test(image_path):
    image = cv2.imread(image_path)
    resized_image = resize_image(image, target_width=1080)
    img = master_preprocessor(resized_image)
    cv2.imshow('master_preprocessor', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test("/Users/om/Desktop/data/low-light.jpg")

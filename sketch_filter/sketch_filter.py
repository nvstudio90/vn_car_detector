import cv2
from util.image_util import resize_image


def sketch_filter(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(grey)
    blurring_image = cv2.GaussianBlur(inverted_image, (21,21), 0)
    inverted_blur_image = 255 - blurring_image
    return cv2.divide(grey, inverted_blur_image, scale=256.0)

if __name__ == '__main__':
    path = "/Users/om/Desktop/data/smiling-asian-young-woman-face-portrait.jpg"
    image_color = cv2.imread(path)
    resized_img = resize_image(image_color, target_width=1080)
    sketch = sketch_filter(resized_img)
    cv2.imshow("sketch", sketch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



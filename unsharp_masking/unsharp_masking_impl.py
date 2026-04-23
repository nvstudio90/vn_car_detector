import cv2
from util.image_util import resize_image


def unsharp_masking_image(original_image):
    blurred = cv2.GaussianBlur(original_image,(5,5),1.0)
    return cv2.addWeighted(original_image,1.5, blurred,-0.5,0.0)

if __name__ == '__main__':
    image = cv2.imread('/Users/om/Desktop/data/test2.webp')
    resizeImg = resize_image(image, target_width=1080)
    cv2.imshow('Original Image', resizeImg)
    newImage = unsharp_masking_image(resizeImg)
    cv2.imshow('Unsharp Mask', newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
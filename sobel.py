import cv2
import numpy as np


def sobel_sample(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    cv2.imshow('sobel_x', sobel_x)
    cv2.imshow('sobel_y', sobel_y)
    cv2.imshow('sobel_combined', sobel_combined / np.max(sobel_combined))
    cv2.waitKey(0)

def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 10, 150)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

if __name__ == '__main__':
    # sobel_sample('/Users/om/Desktop/data/smiling-asian-young-woman-face-portrait.jpg')
   edge_detection('/Users/om/Desktop/data/smiling-asian-young-woman-face-portrait.jpg')

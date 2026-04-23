import cv2
import numpy as np


def show_image():
    path = "/Users/om/Desktop/landmark-forest-tourism-sunrise-famous-ancient.jpg"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        dimension = img.shape
        height = dimension[0]
        width = dimension[1]
        channels = dimension[2] if len(img.shape) > 2 else 1
        total_pixels = img.size
        data_type = img.dtype
        print(f"--- Thông tin ảnh ---")
        print(f"Kích thước (H x W): {height} x {width}")
        print(f"Số kênh màu: {channels}")
        print(f"Tổng số pixel: {total_pixels}")
        print(f"Kiểu dữ liệu: {data_type}")
    cv2.imshow(path, img)
    cv2.waitKey(0)


def averaging_filter():
    path = "/Users/om/Desktop/landmark-forest-tourism-sunrise-famous-ancient.jpg"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    dst = cv2.blur(img, (101, 101))
    cv2.imshow("blur", dst)
    cv2.waitKey(0)


def gaussian_blur():
    path = "/Users/om/Desktop/landmark-forest-tourism-sunrise-famous-ancient.jpg"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    dst = cv2.GaussianBlur(img, (101, 101), 0)
    cv2.imshow("blur", dst)
    cv2.waitKey(0)

def edge_detection():
    img = cv2.imread("/Users/om/Desktop/data/smiling-asian-young-woman-face-portrait.jpg", cv2.IMREAD_GRAYSCALE)
    edges_noisy = cv2.Canny(img, 50, 150)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    edges_clean = cv2.Canny(blurred_img, 50, 150)
    res = np.hstack((edges_noisy, edges_clean))  # Ghép 2 ảnh nằm ngang
    cv2.imshow('Result: Without Blur (Left) vs With Blur (Right)', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rectangle_detection():
    img = cv2.imread('/Users/om/Desktop/data/rec2.webp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y = approx[0][0]
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif len(approx) == 4:
            (x1, y1, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            name = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif len(approx) > 10:  # Nếu quá nhiều đỉnh, khả năng cao là hình tròn
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Detection', img)
    cv2.waitKey(0)

def brightness_balance():
    img = cv2.imread("/Users/om/Desktop/data/low-light.jpg")
    # khu nhieu
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img_lab = cv2.merge((l_clahe, a, b))
    img_dst = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img_dst, -1, kernel)
    res = np.hstack((img, sharpened))
    cv2.imshow("Brightness", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def high_filter():
    img = cv2.imread("/Users/om/Desktop/data/low-light.jpg")
    blurred = cv2.GaussianBlur(img, (101, 101), 0)
    detail = cv2.subtract(img, blurred)
    cv2.imshow('High-pass Detail', detail)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # show_image()
    # averaging_filter()
    # gaussian_blur()
    # edge_detection()
    # rectangle_detection()
    # brightness_balance()
    high_filter()

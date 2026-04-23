import cv2

# cai thien chat luong van ban
def prepare_image(color_image):
    # buoc 1 chuyen doi sang anh xám
    grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # bước 2 khủ nhiễu giữ cạnh
    image = cv2.bilateralFilter(grey_image, 9, 75, 75)
    # bước 3 làm mờ ảnh
    blured_image = cv2.GaussianBlur(image, (5, 5), 0)
    # bước 4 Unsharp Masking
    unsharp_mask_image = cv2.addWeighted(image, 1.5, blured_image, -0.5, 0)
    # bước cuối áp dụng ngưỡng để đưa ảnh về trắng đen
    return cv2.adaptiveThreshold(unsharp_mask_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

if __name__ == '__main__':
    image = cv2.imread("/Users/om/Desktop/data/doc_scan.webp")
    final_image = prepare_image(image)
    cv2.imshow('Prepare Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

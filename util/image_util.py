import cv2

def resize_image(image, target_width=None, target_height=None, interpolation=cv2.INTER_AREA):
    """
    Resize image với tùy chọn giữ tỉ lệ hoặc ép kích thước.
    - interpolation: cv2.INTER_AREA (tốt khi thu nhỏ), cv2.INTER_CUBIC (tốt khi phóng to)
    """
    h, w = image.shape[:2]

    # Trường hợp 1: Nếu không truyền width và height, trả về ảnh gốc
    if target_width is None and target_height is None:
        return image

    # Trường hợp 2: Giữ tỉ lệ theo Width
    if target_height is None:
        ratio = target_width / float(w)
        target_height = int(h * ratio)

    # Trường hợp 3: Giữ tỉ lệ theo Height
    elif target_width is None:
        ratio = target_height / float(h)
        target_width = int(w * ratio)

    # Trường hợp 4: Ép đúng kích thước target_width & target_height (có thể gây méo ảnh)
    # Logic này đã được xử lý bởi việc khởi tạo dsize bên dưới

    dim = (target_width, target_height)

    # Thực hiện resize
    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized

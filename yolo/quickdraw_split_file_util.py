import os
import random
import shutil


def get_all_categories(root_dir_path):
    sub_folders = [f.name for f in os.scandir(root_dir_path) if f.is_dir()]
    for sub in sub_folders:
        print(f"{sub}")
    return sub_folders


def split_image_for_yolo_train():
    # 1. Cấu hình đường dẫn
    SOURCE_DIR = '../data/quickdraw_data'
    # Folder đích sau khi chia
    TARGET_DIR = '../data/quickdraw_dataset_final'

    # Danh sách các chủ đề (tên folder gốc)
    CATEGORIES = get_all_categories(SOURCE_DIR)

    # Tỷ lệ chia
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    # TEST_RATIO còn lại là 0.1

    # 2. Tạo cấu trúc thư mục mới (test, traint, valid)
    splits = ['train', 'valid', 'test']
    sub_dirs = ['images', 'labels']

    for s in splits:
        for sub in sub_dirs:
            os.makedirs(os.path.join(TARGET_DIR, s, sub), exist_ok=True)

    print("Đang bắt đầu chia dữ liệu...")

    # 3. Xử lý từng chủ đề
    for category in CATEGORIES:
        cat_path = os.path.join(SOURCE_DIR, category)
        img_dir = os.path.join(cat_path, 'images')
        lbl_dir = os.path.join(cat_path, 'labels')

        # Kiểm tra folder có tồn tại không
        if not os.path.exists(img_dir):
            print(f"Bỏ qua {category} vì không tìm thấy folder images.")
            continue

        # Lấy danh sách file ảnh (giả sử định dạng .jpg, bạn có thể đổi thành .png nếu cần)
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Trộn ngẫu nhiên để đảm bảo tính khách quan
        random.seed(42)  # Cố định seed để kết quả giống nhau mỗi lần chạy
        random.shuffle(images)

        # Tính toán điểm chia
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        valid_end = train_end + int(total * VALID_RATIO)

        # Chia danh sách file
        train_files = images[:train_end]
        valid_files = images[train_end:valid_end]
        test_files = images[valid_end:]

        dataset_splits = {
            'train': train_files,
            'valid': valid_files,
            'test': test_files
        }

        # 4. Di chuyển hoặc Copy file vào folder mới
        for split_name, file_list in dataset_splits.items():
            print(f"  Đang xử lý {split_name} cho {category}...")
            for filename in file_list:
                # Tên file không có phần mở rộng
                file_stem = os.path.splitext(filename)[0]

                # Đường dẫn file nguồn
                src_img = os.path.join(img_dir, filename)
                src_lbl = os.path.join(lbl_dir, file_stem + '.txt')

                # Đường dẫn đích
                dst_img = os.path.join(TARGET_DIR, split_name, 'images', filename)
                dst_lbl = os.path.join(TARGET_DIR, split_name, 'labels', file_stem + '.txt')

                # Thực hiện copy (Dùng shutil.copy để giữ lại bản gốc, shutil.move nếu muốn dọn dẹp)
                if os.path.exists(src_lbl):
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_lbl, dst_lbl)
                else:
                    print(f"    Cảnh báo: Thiếu file label cho {src_img}")

    print(f"\nHoàn tất! Dataset đã được chia tại: {os.path.abspath(TARGET_DIR)}")


if __name__ == '__main__':
    root_dir = "../data/quickdraw_data"
    # get_all_categories(root_dir)
    split_image_for_yolo_train()

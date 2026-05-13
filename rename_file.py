import os

def rename_files_in_folder(folder_path):
    try:
        # Lấy danh sách tất cả các mục trong thư mục và sắp xếp theo tên
        items = os.listdir(folder_path)
        items.sort()

        index = 306
        for item in items:
            # Lấy đường dẫn tuyệt đối của mục hiện tại
            old_filepath = os.path.join(folder_path, item)

            # Kiểm tra xem nó có phải là file không (bỏ qua các thư mục con)
            if os.path.isfile(old_filepath):
                # Tách tên file và đuôi mở rộng (vd: từ 'photo.jpg' lấy ra '.jpg')
                file_extension = os.path.splitext(item)[1]

                # Cấu trúc tên mới: img_1.jpg, img_2.png...
                new_filename = f"img_{index}{file_extension}"
                new_filepath = os.path.join(folder_path, new_filename)

                # Thực hiện đổi tên
                os.rename(old_filepath, new_filepath)
                print(f"Đã đổi tên: {item} -> {new_filename}")

                index += 1

        print("\n✅ Hoàn tất quá trình đổi tên!")

    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy thư mục '{folder_path}'")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi: {e}")

if __name__ == '__main__':
    # --- CÁCH SỬ DỤNG ---
    # Thay thế đường dẫn dưới đây bằng đường dẫn tới thư mục của bạn.
    # Dùng r"..." (raw string) trên Windows để tránh lỗi dấu gạch chéo ngược.
    target_folder = "data/dataset/car_plates2"

    rename_files_in_folder(target_folder)
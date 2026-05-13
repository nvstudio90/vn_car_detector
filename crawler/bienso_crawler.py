from icrawler.builtin import GoogleImageCrawler
import os
import requests
from ddgs import DDGS
import time

def download_license_plates(keyword, folder_name, max_num=50):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"--- Đang tải ảnh cho từ khóa: {keyword} ---")

    # Khởi tạo bộ crawl của Google
    google_crawler = GoogleImageCrawler(storage={'root_dir': folder_name})

    # Thực hiện tìm kiếm và tải ảnh
    google_crawler.crawl(keyword=keyword, max_num=max_num)


def download_images(query, folder, max_images=50):
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f"--- Đang tìm kiếm ảnh cho: {query} ---")

    with DDGS() as ddgs:
        # Lấy danh sách URL ảnh từ DuckDuckGo
        results = ddgs.images(
            query=query,
            region="wt-wt",
            safesearch="off",
            size="Medium",
            max_results=max_images
        )

        count = 0
        for i, res in enumerate(results):
            if count >= max_images:
                break

            image_url = res['image']
            try:
                # Tải ảnh về
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    # Lấy đuôi file (jpg, png...)
                    ext = image_url.split('.')[-1].split('?')[0]
                    if ext.lower() not in ['jpg', 'jpeg', 'png']:
                        ext = 'jpg'

                    file_path = os.path.join(folder, f"img_{count + 1}.{ext}")
                    with open(file_path, 'wb') as f:
                        f.write(response.content)

                    print(f"Đã tải: {file_path}")
                    count += 1
                    # Nghỉ một chút để tránh bị chặn
                    time.sleep(0.5)
            except Exception as e:
                print(f"Lỗi khi tải ảnh {i}: {e}")


if __name__ == "__main__":
    # Tải 50 ảnh biển số ô tô
    download_images("biển số xe ô tô việt nam", "../data/dataset/car_plates2", 100)

    # Tải 50 ảnh biển số xe máy
    download_images("biển số xe máy việt nam", "../data/dataset/motorbike_plates2", 100)

    print("\n--- Hoàn thành! Kiểm tra thư mục 'dataset' nhé. ---")


# if __name__ == "__main__":
#     # 1. Tải ảnh biển số ô tô Việt Nam
#     download_license_plates(
#         keyword='biển số xe ô tô việt nam mới nhất',
#         folder_name='../data/dataset/car_plates',
#         max_num=50
#     )
#
#     # 2. Tải ảnh biển số xe máy Việt Nam
#     download_license_plates(
#         keyword='biển số xe máy việt nam',
#         folder_name='../data/dataset/motorbike_plates',
#         max_num=50
#     )
#
#     print("\nHoàn thành! Ảnh đã được lưu trong thư mục 'dataset'.")
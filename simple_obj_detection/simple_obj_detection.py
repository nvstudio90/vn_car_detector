import cv2
import numpy as np

def simple_obj_detection(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    previous_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended")
            break
        if previous_frame is None:
            previous_frame = frame
        else :
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey_previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            blur_frame = cv2.GaussianBlur(grey_frame, (9,9), 0)
            blur_previous_frame = cv2.GaussianBlur(grey_previous_frame, (9,9), 0)
            diff = cv2.absdiff(blur_frame, blur_previous_frame)
            # phân ngưỡng
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            # Bước 4: Làm sạch bằng Morphological
            # Xóa nhiễu li ti (Erode) sau đó làm dày vật thể (Dilate)
            kernel = np.ones((3,3),np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            result = cv2.bitwise_and(grey_frame, grey_frame, mask=cleaned)
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("video_frame", frame)
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1000 / fps
            if cv2.waitKey(int(delay)) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = "../data/test.MOV"
    simple_obj_detection(image_path)





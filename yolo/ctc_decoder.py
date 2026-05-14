import numpy as np

class PureCTCDecoder:
    def __init__(self, dict_path):
        # 1. Nạp từ điển từ file text
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.character_dict = [line.strip("\n\r") for line in f.readlines()]
        # PaddleOCR thường tự động thêm ký tự 'SPACE' vào từ điển
        self.character_dict.append(' ')
        self.blank_idx = 0

    def decode(self, preds):
        # preds là kết quả từ OpenVINO, shape thường là [1, số_cột, số_class] -> ví dụ [1, 40, 838]
        preds = preds[0]  # Bỏ qua chiều batch size

        # Tìm index có xác suất lớn nhất tại mỗi time-step
        pred_indices = np.argmax(preds, axis=1)
        pred_probs = np.max(preds, axis=1)

        text = ""
        conf_list = []
        for i in range(len(pred_indices)):
            idx = pred_indices[i]
            # Bỏ qua Blank và các ký tự lặp liên tiếp
            if idx != self.blank_idx and (i == 0 or idx != pred_indices[i - 1]):
                char_idx = idx - 1
                # Đảm bảo index nằm trong giới hạn an toàn
                if 0 <= char_idx < len(self.character_dict):
                    text += self.character_dict[char_idx]
                    conf_list.append(pred_probs[i])

        # Tính độ chính xác trung bình
        mean_conf = np.mean(conf_list) if len(conf_list) > 0 else 0.0
        if mean_conf > 0.8:
            return text, mean_conf
        return "", 0.0
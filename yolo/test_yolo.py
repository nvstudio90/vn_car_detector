from ultralytics import YOLO

def train_data():
    model = YOLO("yolo26s.pt")
    model.train(data = "../data/TestCarModel.v1i.yolo26/data.yaml", epochs = 50,
                name = "YOLO_BienSo_V1",
                project = "trained")
if __name__ == '__main__':
    train_data()
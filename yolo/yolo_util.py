from ultralytics import YOLO

def export_to_vino(model_path):
    yolo = YOLO(model_path)
    yolo.export(format = "openvino", imgsz=640, half=True)

export_to_vino("../data/model_trained/vietnam_car_nano_model.pt")

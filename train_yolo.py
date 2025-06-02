from ultralytics import YOLO

# Załaduj model segmentacyjny (możesz użyć 'yolov8n-seg.pt', 'yolov8s-seg.pt' itd.)
model = YOLO("yolov8n-seg.pt")

# Trenuj
model.train(
    data="C:/Users/Damian/OneDrive - Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie/Pulpit/rotter/dataset/data.yaml",  # Ścieżka do pliku YAML
    epochs=50,
    imgsz=640,
    batch=8,
    name="robot-segmentation",
    project="runs3",  # katalog z wynikami
)
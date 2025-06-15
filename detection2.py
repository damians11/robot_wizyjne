import cv2
from ultralytics import YOLO
import numpy as np
import time

# === 1. Załaduj wytrenowany model ===
model = YOLO("runs/robot-segmentation/weights/best.pt")

# === 2. Uruchom kamerę ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

# Czas ostatniego wypisu do konsoli
last_print_time = time.time()

# === 3. Przetwarzanie klatek ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.6, verbose=False)[0]
    object_positions = {}

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        names = results.names
        classes = results.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):
            cls_id = int(classes[i])
            label = names[cls_id]

            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    object_positions.setdefault(label, []).append((cx, cy))
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Wypisuj co 2 sekundy
    current_time = time.time()
    if current_time - last_print_time >= 2.0:
        print("🕒 Obiekty na scenie:")
        for label, positions in object_positions.items():
            for (x, y) in positions:
                print(f"  {label}: ({x}, {y})")
        last_print_time = current_time

    # === 4. Wyświetl obraz ===
    cv2.imshow("Wykrywanie w czasie rzeczywistym", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

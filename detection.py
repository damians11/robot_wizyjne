import cv2
from ultralytics import YOLO
import numpy as np

# === 1. Załaduj wytrenowany model ===
model = YOLO("runs/robot-segmentation/weights/best.pt")  # Ścieżka do modelu

# === 2. Uruchom kamerę ===
cap = cv2.VideoCapture(0)  # 0 = domyślna kamera
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

# === 3. Przetwarzanie klatek ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.6, verbose=False)[0]

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        names = results.names
        classes = results.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):
            cls_id = int(classes[i])
            label = names[cls_id]

            # Przetwarzanie maski na kontur
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Rysuj kontur
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                # Oblicz środek konturu
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # === 4. Pokaż wynik ===
    cv2.imshow("Wykrywanie w czasie rzeczywistym", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC = wyjście
        break

cap.release()
cv2.destroyAllWindows()
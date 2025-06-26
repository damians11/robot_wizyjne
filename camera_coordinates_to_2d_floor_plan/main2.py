import cv2
import numpy as np

# === KONFIGURACJA ===
url = "http://192.168.0.194:8080/video"  # Adres strumienia z telefonu
MAP_WIDTH = 1000
MAP_HEIGHT = 1000

# Ustal punkty narożne (np. ręcznie wcześniej dobrane na statycznym zdjęciu)
# Kolejność: top-left, top-right, bottom-right, bottom-left
points_array = np.load("selected_points.npy")
reference_points = [tuple(pt) for pt in points_array.tolist()]

# Oblicz macierz homografii
dst_points = np.array([
    [0, 0],
    [MAP_WIDTH - 1, 0],
    [MAP_WIDTH - 1, MAP_HEIGHT - 1],
    [0, MAP_HEIGHT - 1]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(np.array(reference_points, dtype=np.float32), dst_points)


def transform_frame(frame, H, width, height):
    """Apply perspective transformation using homography matrix."""
    return cv2.warpPerspective(frame, H, (width, height))


def main():
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Nie można otworzyć strumienia z kamery.")
        return

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        if not ret:
            print("Nie udało się odczytać ramki z kamery.")
            break

        # Resize if needed for performance (opcjonalnie)
        

        # Przekształcony widok
        transformed = transform_frame(frame, H, MAP_WIDTH, MAP_HEIGHT)

        # Pokaż oba okna
        cv2.imshow("Widok z kamery (oryginalny)", frame)
        cv2.imshow("Widok 2D (rzut z góry)", transformed)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC kończy działanie
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

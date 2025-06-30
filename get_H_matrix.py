import cv2
import numpy as np

# === Konfiguracja ===
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
CAMERA_SOURCE = 0#"http://192.168.0.194:8080/video"  # Możesz też użyć 0 dla lokalnej kamery

labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
instructions = [
    "Wybierz punkt 1: top-left",
    "Wybierz punkt 2: top-right",
    "Wybierz punkt 3: bottom-right",
    "Wybierz punkt 4: bottom-left"
]

points = []
done = False

def mouse_callback(event, x, y, flags, param):
    global points, done
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Punkt {len(points)} ({labels[len(points)-1]}): {x}, {y}")
        if len(points) == 4:
            done = True

def main():
    global done

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    cv2.namedWindow("Wybierz 4 punkty (ESC aby anulować)")
    cv2.setMouseCallback("Wybierz 4 punkty (ESC aby anulować)", mouse_callback)

    while True:
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            break

        display = frame.copy()

        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, f"{i+1} ({labels[i]})", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(points) < 4:
            cv2.putText(display, instructions[len(points)], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Wybierz 4 punkty (ESC aby anulować)", display)

        if done:
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("Anulowano wybór punktów.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # Liczenie i zapis homografii
    src_points = np.array(points, dtype='float32')
    dst_points = np.array([
        [0, 0],
        [MAP_WIDTH - 1, 0],
        [MAP_WIDTH - 1, MAP_HEIGHT - 1],
        [0, MAP_HEIGHT - 1]
    ], dtype='float32')

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    np.save("homography_matrix.npy", H)
    np.save("selected_points.npy", src_points)

    print("Macierz homografii zapisana do homography_matrix.npy")
    print("Macierz H:")
    print(H)

if __name__ == "__main__":
    main()

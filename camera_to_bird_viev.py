import cv2
import numpy as np

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[2] = pts[np.argmax(diff)]   # bottom-left
    rect[3] = pts[np.argmax(s)]      # bottom-right

    return rect

def select_points(image):
    points = []
    instructions = [
        "Zaznacz punkt 1 (górny-lewy)",
        "Zaznacz punkt 2 (górny-prawy)",
        "Zaznacz punkt 3 (dolny-lewy)",
        "Zaznacz punkt 4 (dolny-prawy)"
    ]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Wybrano punkt: {x}, {y}")

    clone = image.copy()
    cv2.namedWindow("Kliknij 4 punkty")
    cv2.setMouseCallback("Kliknij 4 punkty", mouse_callback)

    while True:
        temp = clone.copy()
        if len(points) < 4:
            cv2.putText(temp, instructions[len(points)], (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        for idx, p in enumerate(points):
            cv2.circle(temp, p, 5, (0, 0, 255), -1)
            cv2.putText(temp, f"{idx + 1}", (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Kliknij 4 punkty", temp)

        if len(points) == 4:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            print("Przerwano wybór punktów.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return points

def detect_floor_auto(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Szukaj największego konturu
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            pts = [tuple(pt[0]) for pt in approx]
            return order_points(pts)

    return None

def warp_image(image, pts_src):
    width = int(max(
        np.linalg.norm(pts_src[0] - pts_src[1]),
        np.linalg.norm(pts_src[2] - pts_src[3])
    ))
    height = int(max(
        np.linalg.norm(pts_src[0] - pts_src[2]),
        np.linalg.norm(pts_src[1] - pts_src[3])
    ))

    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

def main():
    mode = "manual"  # <- zmień na "manual", jeśli chcesz klikać ręcznie

    image_path = 'dataset/valid/images/IMG_1254_JPG.rf.3e5d069b50a45684bfe88db34b2c5173.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd: nie można otworzyć pliku '{image_path}'")
        return

    if mode == "manual":
        points = select_points(image)
        if points is None or len(points) != 4:
            print("Nie wybrano poprawnych 4 punktów.")
            return
        pts_src = order_points(points)

    elif mode == "auto":
        pts_src = detect_floor_auto(image)
        if pts_src is None:
            print("Nie udało się wykryć podłoża automatycznie.")
            return

    else:
        print("Nieznany tryb działania.")
        return

    warped = warp_image(image, pts_src)
    cv2.imshow("Rzut z góry", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

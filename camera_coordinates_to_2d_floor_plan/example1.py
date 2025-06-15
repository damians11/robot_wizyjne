import cv2
import numpy as np

# --------- USTAWIENIA ---------
img_path = "dataset/valid/images/IMG_1254_JPG.rf.3e5d069b50a45684bfe88db34b2c5173.jpg"  # <- ZMIEŃ TUTAJ ŚCIEŻKĘ DO PLIKU
map_size = 1000  # mapa 2D będzie 1000x1000 pikseli
# ------------------------------

# Wczytaj obraz
img = cv2.imread(img_path)
if img is None:
    print("Nie można wczytać obrazu. Sprawdź ścieżkę.")
    exit()

# Lista klikniętych punktów
clicked_points = []

# Funkcja obsługująca kliknięcia
def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Punkt {len(clicked_points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Obraz", img)

# Krok 1: kliknij 4 punkty narożne
print("Kliknij 4 punkty narożne (zgodnie z ruchem wskazówek zegara, zaczynając od lewego górnego).")
cv2.imshow("Obraz", img)
cv2.setMouseCallback("Obraz", click_event)
cv2.waitKey(0)

if len(clicked_points) != 4:
    print("Musisz kliknąć dokładnie 4 punkty.")
    exit()

pts_src = np.array(clicked_points, dtype="float32")
pts_dst = np.array([
    [0, 0],
    [map_size, 0],
    [map_size, map_size],
    [0, map_size]
], dtype="float32")

# Oblicz homografię
H, _ = cv2.findHomography(pts_src, pts_dst)
print("Macierz homografii:\n", H)

# Krok 2: klikaj inne punkty do przekształcenia
clicked_points = []
print("\nKlikaj inne punkty, które chcesz przekształcić na mapę 2D.")
print("Wciśnij dowolny klawisz, by zakończyć.")

def transform_and_show(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        punkt = np.array([[[x, y]]], dtype="float32")
        wynik = cv2.perspectiveTransform(punkt, H)
        x_map, y_map = wynik[0][0]
        print(f"Punkt ({x}, {y}) -> Mapa 2D: ({x_map:.2f}, {y_map:.2f})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Obraz", img)

cv2.setMouseCallback("Obraz", transform_and_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

import torch
import cv2
import numpy as np
import os

def load_midas_model(device):
    model_type = "DPT_Large"  # Możesz zmienić na "DPT_Hybrid" lub "MiDaS_small"
    
    # Załaduj model MiDaS
    midas = torch.hub.load("isl-org/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Załaduj odpowiednie transformaty dla danego typu modelu
    transforms = torch.hub.load("isl-org/MiDaS", "transforms")

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform

def estimate_depth(image_path, device):
    midas, transform = load_midas_model(device)

    # Wczytaj obraz z pliku
    img = cv2.imread(image_path)
    if img is None:
        print("Błąd: nie można otworzyć obrazu.")
        return

    # Konwersja BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Przygotuj obraz do sieci
    input_batch = transform(img_rgb).to(device)


    print(f"Input batch shape: {input_batch.shape}")  # np. [1, 3, 384, 384]

    # Oblicz głębię
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],  # Dopasuj do oryginalnych wymiarów obrazu
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalizacja do zakresu 0–255
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_map * 255).astype(np.uint8)

    # Kolorowanie mapy głębokości
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # Wyświetlenie
    cv2.imshow("Depth Map", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Opcjonalnie zapisz wynik
    output_path = "depth_output.png"
    cv2.imwrite(output_path, depth_colored)
    print(f"Zapisano: {output_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "dataset/valid/images/IMG_1254_JPG.rf.3e5d069b50a45684bfe88db34b2c5173.jpg"
    estimate_depth(image_path, device)

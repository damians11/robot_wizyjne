import torch
import urllib.request
import cv2
import numpy as np
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def load_midas_model(device):
    model_type = "DPT_Large"  # lub "DPT_Hybrid", "MiDaS_small"
    midas = torch.hub.load("isl-org/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform if model_type.startswith("DPT") else torch.hub.load("isl-org/MiDaS", "transforms").small_transform
    return midas, transform

def estimate_depth(image_path, device):
    midas, transform = load_midas_model(device)

    img = cv2.imread(image_path)
    if img is None:
        print("Błąd: nie można otworzyć obrazu.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalizacja do zakresu 0–255
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    depth_uint8 = (depth_map * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    cv2.imshow("Depth Map", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimate_depth("dataset/valid/images/IMG_1254_JPG.rf.3e5d069b50a45684bfe88db34b2c5173.jpg", device)

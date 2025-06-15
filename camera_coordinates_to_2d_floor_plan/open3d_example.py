import torch
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

def load_midas_model(device):
    model_type = "DPT_Large"
    midas = torch.hub.load("isl-org/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    transform = transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else transforms.small_transform
    return midas, transform

def estimate_depth(image_path, device):
    midas, transform = load_midas_model(device)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Nie można otworzyć obrazu.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    # Normalizacja i przeskalowanie do mm
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map = depth_map * 1000.0
    return depth_map, img

def depth_to_point_cloud(depth_map, fx=500.0, fy=500.0, cx=None, cy=None):
    h, w = depth_map.shape
    cx = cx if cx is not None else w / 2
    cy = cy if cy is not None else h / 2

    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth_map
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1)
    return xyz.reshape(-1, 3), j.reshape(-1), i.reshape(-1)  # x,y to kolumny,wiersze

def detect_ground_plane(depth_map):
    points, rows, cols = depth_to_point_cloud(depth_map)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel_size=5.0)
    pcd.remove_non_finite_points()

    plane_model, inliers = pcd.segment_plane(distance_threshold=5.0,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Płaszczyzna: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    ground = pcd.select_by_index(inliers)
    return ground, plane_model

def visualize_ground_2d(ground):
    points_np = np.asarray(ground.points)
    points_np *= 0.001
    points_2d = points_np[:, [0, 2]]

    plt.figure(figsize=(8, 6))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, label="punkty")

    if len(points_2d) >= 3:
        hull = ConvexHull(points_2d)
        for simplex in hull.simplices:
            plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'r-')

    plt.title("Rzut 2D - wykryte podłoże (XZ)")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def draw_ground_on_image(img, depth_map, ground):
    h, w = depth_map.shape
    all_points, rows, cols = depth_to_point_cloud(depth_map)
    ground_points = np.asarray(ground.points)

    # Dopasuj KDTree do pełnej chmury
    kdt = KDTree(all_points)
    _, indices = kdt.query(ground_points, k=1)
    mask_flat = np.zeros((h * w,), dtype=np.uint8)
    mask_flat[indices.flatten()] = 255
    mask = mask_flat.reshape((h, w))

    img_masked = img.copy()
    img_masked[mask > 0] = [0, 255, 0]

    cv2.imshow("Obraz z zaznaczonym podłożem", img_masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "dataset/valid/images/IMG_1254_JPG.rf.3e5d069b50a45684bfe88db34b2c5173.jpg"

    depth_map, img = estimate_depth(image_path, device)
    ground, _ = detect_ground_plane(depth_map)
    visualize_ground_2d(ground)
    draw_ground_on_image(img, depth_map, ground)

if __name__ == "__main__":
    main()

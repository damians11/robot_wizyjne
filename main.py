import cv2
from ultralytics import YOLO
import numpy as np
import time
from control_alg import move_robot
from visualization import init_view, step
import paho.mqtt.client as mqtt  

# === KONFIGURACJA ===
url = 1#"http://192.168.0.194:8080/video"
model = YOLO("runs/robot-segmentation/weights/best.pt")
MAP_SIZE = 1000

# === MQTT ===
BROKER_ADDR = "192.168.0.101"
BROKER_PORT = 1883
MQTT_TOPIC  = "robot/command"

def init_mqtt() -> mqtt.Client:
    cli = mqtt.Client()
    cli.connect(BROKER_ADDR, BROKER_PORT, keepalive=60)
    cli.loop_start()
    print(f"[MQTT] Połączono z brokerem {BROKER_ADDR}:{BROKER_PORT}")
    return cli

# === Wczytaj macierz homografii ===
H = np.load("homography_matrix.npy")
print("Wczytano macierz homografii:\n", H)

# === Uruchom kamerę ===
cap = cv2.VideoCapture(url)

mqtt_cli = init_mqtt()
last_print_time = time.time()

def calculate_new_coordinate(x, y, H):
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    new_x, new_y = transformed[0][0]
    return int(round(new_x)), int(round(new_y))

def draw_on_map(map_size, mapped_positions):
    map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    for label, positions in mapped_positions.items():
        for x, y in positions:
            if 0 <= x < map_size and 0 <= y < map_size:  # <== DODAJ TO!
                cv2.circle(map_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(map_img, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return map_img


def transform_frame(frame, H, width, height):
    """Apply perspective transformation using homography matrix."""
    return cv2.warpPerspective(frame, H, (width, height))

def main():
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    global last_print_time

    init_view()
    robot_pos = [0,0]
    meta_pos = []
    obstacles = [[0,0],[0,0],[0,0]]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        transformed_view = transform_frame(frame, H, MAP_SIZE, MAP_SIZE)

        results = model.predict(source=frame, conf=0.6, verbose=False)[0]
        object_positions = {}
        mapped_positions = {}

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
                        mx, my = calculate_new_coordinate(cx, cy, H)
                        mapped_positions.setdefault(label, []).append((mx, my))

                        # Oznaczenia na oryginalnym obrazie
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # Oznaczenia na przekształconym widoku
                        if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                            cv2.circle(transformed_view, (mx, my), 5, (0, 0, 255), -1)
                            cv2.putText(transformed_view, label, (mx + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        idx = 0
        # Wypisuj co 2 sekundy
        current_time = time.time()
        if current_time - last_print_time >= 0.1:
            # print("TIME: ", current_time)
            
            # print("Obiekty na scenie:")
            for label, positions in object_positions.items():
                for (x, y) in positions:
                    # print(f"  {label}: ({x}, {y}) → {calculate_new_coordinate(x, y, H)}")
                    if label == "meta":
                        meta_pos = [x,y]
                    elif label == "robot":
                        robot_pos = [x,y]
                    elif label == "przeszkoda" and idx < 3:
                        obstacles[idx] = [x,y]
                        idx = idx+1
            last_print_time = current_time


        # Sterowanie
            if robot_pos != [] and meta_pos != [] and obstacles!= []:
                robot_cmd = move_robot(robot_pos, meta_pos, obstacles)
                if robot_cmd == None:
                    robot_cmd = "stop"
                mqtt_cli.publish(MQTT_TOPIC, robot_cmd)
                print(robot_cmd)

                step(robot_pos, meta_pos, obstacles)
            # else:
                # mqtt_cli.publish(MQTT_TOPIC, "stop")

            robot_pos = []
        
        # Wyświetl obraz kamery, mapę i przekształcony widok
        map_view = draw_on_map(MAP_SIZE, mapped_positions)
        cv2.imshow("Wykrywanie w czasie rzeczywistym", frame)
        cv2.imshow("Top-Down Map", map_view)
        cv2.imshow("Widok 2D z obiektami", transformed_view)

        # Parametry zapisu
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # lub 'XVID' dla .avi

        # Ustal rozmiar w zależności od rozmiaru klatki
        # frame_height, frame_width = frame.shape[:2]
        # map_height, map_width = map_view.shape[:2]
        # trans_height, trans_width = transformed_view.shape[:2]

        # # Inicjalizacja VideoWriter
        # out_frame = cv2.VideoWriter('frame_view.mp4', fourcc, 20.0, (frame_width, frame_height))
        # out_map = cv2.VideoWriter('map_view.mp4', fourcc, 20.0, (map_width, map_height))
        # out_trans = cv2.VideoWriter('transformed_view.mp4', fourcc, 20.0, (trans_width, trans_height))

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            mqtt_cli.publish(MQTT_TOPIC, "stop")
            break

    cap.release()
    mqtt_cli.loop_stop()
    mqtt_cli.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
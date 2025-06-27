import time
import math
from visualization import init_view, step, PROMIEN, ROBOT_HEIGHT, ROBOT_WIDTH, y

# Parametry robota - UZUPEŁNIJ
# ROBOT_WIDTH = 10
# ROBOT_HEIGHT = 6
MAX_LINE_FOLLOW = 15 # max kroków do obchodzenia przeszkód
SAFE_DISTANCE = 2  # dodatkowa przestrzeń bezpieczeństwa
#____________________________________________________________________________________
ROBOT_RADIUS = max(ROBOT_WIDTH, ROBOT_HEIGHT) / 2


# Funkcja pomocnicza: dystans euklidesowy
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Funkcja pomocnicza: sprawdzenie kolizji z przeszkodami
def check_collision(robot_pos, obstacles):
    for obs in obstacles:
        if distance(robot_pos, obs) < (ROBOT_RADIUS + PROMIEN + SAFE_DISTANCE):
            return True
    return False

# Funkcja ruchu w stronę mety z unikanie kolizji
def move_robot(robot_pos, meta_pos, obstacles, step_size=1.0):
    dx = meta_pos[0] - robot_pos[0]
    dy = meta_pos[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    dist = math.hypot(dx, dy)

    if dist < 1e-2:
        return robot_pos  # już na miejscu

def move_robot(robot_pos, meta_pos, obstacles, step_size=1.0):
    if distance(robot_pos, meta_pos) < ROBOT_RADIUS:
        print("Dotarto do mety!")
        return "stop"
    
    dx = meta_pos[0] - robot_pos[0]
    dy = meta_pos[1] - robot_pos[1]
    current_dist = math.hypot(dx, dy)

    # Zdefiniuj możliwe kierunki jako wektory
    directions = {
        "forward": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "backward": (-1, 0),
    }

    # Inicjalizacja zmiennych stanu
    if not hasattr(move_robot, "wall_follow_mode"):
        move_robot.wall_follow_mode = False
    if not hasattr(move_robot, "line_follow_steps"):
        move_robot.line_follow_steps = 0

    # Posortuj kierunki według przybliżenia do mety (najpierw te, które zmniejszają dystans)
    def estimated_distance(dir_vec):
        new_x = robot_pos[0] + dir_vec[0] * step_size
        new_y = robot_pos[1] + dir_vec[1] * step_size
        return math.hypot(meta_pos[0] - new_x, meta_pos[1] - new_y)

    # sorted_dirs = sorted(directions.items(), key=lambda item: estimated_distance(item[1]))

    # for name, (dx_step, dy_step) in sorted_dirs:
    #     new_pos = [robot_pos[0] + dx_step * step_size, robot_pos[1] + dy_step * step_size]
    #     if not check_collision(new_pos, obstacles):
    #         return name  # zwróć pierwszy bezpieczny ruch

     # --- LINE FOLLOW (kierunek do mety) ---
    # --- TRYB: LINE FOLLOW ---
    # print(move_robot.wall_follow_mode)
    destination=["backward", "right"]
    if dx>=0:
        destination[0] = "forward"
    else:
        destination[0] = "backward"
    if dy>=0:
        destination[1] = "right"
    else: 
        destination[1] = "left"

    if not move_robot.wall_follow_mode:
        sorted_dirs = sorted(directions.items(), key=lambda item: estimated_distance(item[1]))
        for name, (dx_step, dy_step) in sorted_dirs:
            new_pos = [robot_pos[0] + dx_step * step_size, robot_pos[1] + dy_step * step_size]
            new_dist = estimated_distance((dx_step, dy_step))
            if not check_collision(new_pos, obstacles) and name in destination :
                move_robot.line_follow_steps += 1
                return name

        # Jeśli brak bezpiecznych ruchów zbliżających do celu → wall-follow
        move_robot.wall_follow_mode = True
        move_robot.line_follow_steps = 0

    
    # --- TRYB: WALL FOLLOW ---
    for name in directions:
        dx_step, dy_step = directions[name]
        new_pos = [robot_pos[0] + dx_step * step_size, robot_pos[1] + dy_step * step_size]
        new_dist = estimated_distance((dx_step, dy_step))

        if not check_collision(new_pos, obstacles):
            # Jeśli ruch poprawia dystans – może wyjść z wall-follow
            move_robot.line_follow_steps += 1
            if name in destination or move_robot.line_follow_steps > MAX_LINE_FOLLOW:
                move_robot.wall_follow_mode = False
            # else:
            #     move_robot.line_follow_steps = 0
            return name

    # Brak możliwości ruchu
    return None

def update_pos(robot_pos, dx, dy):
    new_pos = [robot_pos[0]+dx, robot_pos[1]+dy]
    return new_pos


# Główna pętla sterowania
if __name__ == '__main__':
    robot_pos, meta_pos, obstacles = init_view()

    while(1):  
        robot_cmd = move_robot(robot_pos, meta_pos, obstacles) #1
        dx=0
        dy=0
        if robot_cmd=="forward":
            dx = 1
        elif robot_cmd=="left":
            dy=-1
        elif robot_cmd=="right":
            dy=1
        elif robot_cmd=="backward":
            dx=-1
        robot_pos = update_pos(robot_pos, dx, dy)
        step(robot_pos, meta_pos, obstacles) #2

        if distance(robot_pos, meta_pos) < ROBOT_RADIUS:
            print("Dotarto do mety!")
            break

        time.sleep(0.05)

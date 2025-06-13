import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Globalna zmienna określająca promień punktów i wymiary robota
PROMIEN = 3
PROMIEN_META = 2
ROBOT_WIDTH = 10
ROBOT_HEIGHT = 6

# Wymiary obszaru roboczego
x = 200
y = 100

# Inicjalne pozycje
robot = [20, 30]
meta = [80, 70]
przeszkody = [[50, 52], [30, 80], [60, 20], [75, 40], [65, 50]]

# Tryb interaktywny
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, x)
ax.set_ylim(0, y)
ax.set_aspect('equal')
ax.set_facecolor('white')
ax.axis('off')

# Czerwona obwódka
obwodka = patches.Rectangle((0, 0), x, y, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(obwodka)

# Robot jako prostokąt (środek robota ustawiamy jako środek prostokąta)
robot_rect = patches.Rectangle(
    (robot[0] - ROBOT_WIDTH / 2, robot[1] - ROBOT_HEIGHT / 2),
    ROBOT_WIDTH, ROBOT_HEIGHT,
    color='blue'
)
ax.add_patch(robot_rect)

# Meta jako koło
meta_circle = plt.Circle(meta, PROMIEN_META, color='green')
ax.add_patch(meta_circle)

# Przeszkody jako koła
obstacle_circles = [plt.Circle(p, PROMIEN, color='black') for p in przeszkody]
for circle in obstacle_circles:
    ax.add_patch(circle)

plt.draw()
plt.pause(0.001)

# 🔄 Funkcja do aktualizacji pozycji
def update_view(new_robot, new_meta, new_przeszkody):
    # Aktualizacja pozycji prostokąta robota
    robot_rect.set_xy((new_robot[0] - ROBOT_WIDTH / 2, new_robot[1] - ROBOT_HEIGHT / 2))

    # Meta jako koło
    meta_circle.center = new_meta

    # Usuwamy stare przeszkody
    for c in obstacle_circles:
        c.remove()
    obstacle_circles.clear()

    # Rysujemy nowe przeszkody
    for p in new_przeszkody:
        circle = plt.Circle(p, PROMIEN, color='black')
        ax.add_patch(circle)
        obstacle_circles.append(circle)

    # Odświeżenie widoku
    plt.draw()
    plt.pause(0.001)


def init_view():
    return robot, meta, przeszkody

def step(robot_pos, meta_pos, przeszkody_pos):
    update_view(robot_pos, meta_pos, przeszkody_pos)
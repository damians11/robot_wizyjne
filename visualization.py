import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cv2

# UZUPEŁNIJ
PROMIEN =10#3
PROMIEN_META = 10#2
ROBOT_WIDTH = 40#10
ROBOT_HEIGHT = 20#6
# Wymiary obszaru roboczego
x = 1000#200
y = 1000#100
MAX_LINE_FOLLOW = 15 # max kroków do obchodzenia przeszkód
SAFE_DISTANCE = 5  # dodatkowa przestrzeń bezpieczeństwa
#__________________________________________________________

# Inicjalne pozycje
robot = [20, 30]
meta = [80, 70]
przeszkody = [[50, 52]]

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
    # Render jednej klatki, aby uzyskać wymiary
    plt.savefig("temp_plot.png")
    plot_frame = cv2.imread("temp_plot.png")
    plot_height, plot_width = plot_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # lub 'XVID'
    out_plot = cv2.VideoWriter('plot_view.mp4', fourcc, 20.0, (plot_width, plot_height))
    
    plt.pause(0.001)



def init_view():
    return robot, meta, przeszkody

def step(robot_pos, meta_pos, przeszkody_pos):
    pos_tym = [robot_pos[0], 1000-robot_pos[1]]
    meta_tym = [meta_pos[0], 1000-meta_pos[1]]
    przeszkody_tym = [[],[],[]]
    idx = 0
    for el in przeszkody_pos:
         przeszkody_tym[idx] = [przeszkody_pos[idx][0], 1000-przeszkody_pos[idx][1]]
         idx = idx+1
    update_view(pos_tym, meta_tym, przeszkody_tym)
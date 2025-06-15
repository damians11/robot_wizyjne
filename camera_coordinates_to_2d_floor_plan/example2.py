import cv2
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = 'chess.png'


chess = cv2.imread('chess.png')
chess = cv2.cvtColor(chess, cv2.COLOR_BGR2RGB)  # konwersja z BGR do RGB

plt.imshow(chess)
plt.title("Original image")
plt.axis('off')  # opcjonalnie, żeby usunąć osie
plt.show()

#source coordinates
src = np.array([391, 100, 
                14, 271,
                347, 624,
                747, 298,]).reshape((4, 2))
#destination coordinates
dst = np.array([0, 0, 
                0, 600,
                600, 600,
                600, 0,]).reshape((4, 2))


#using skimage’s transform module where ‘projective’ is our desired parameter
tform = transform.estimate_transform('projective', src, dst)
tf_img = transform.warp(chess, tform.inverse)


# plotting the transformed image
fig, ax = plt.subplots()
ax.imshow(tf_img)
ax.set_title('Projective transformation')
ax.axis('off')  # opcjonalnie
plt.show()
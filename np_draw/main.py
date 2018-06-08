import numpy as np
import matplotlib.pyplot as plt

from np_draw.sprite import Sprite

cross_img = np.zeros((30, 30))
cross_img[13:16, :] = 1
cross_img[:, 13:16] = 1

cross_sprite = Sprite(cross_img, cross_img>0, np.array([30, 16]), 75)
cross_sprite2 = Sprite(cross_img, cross_img>0, np.array([25, 35]), 0)

canvas = np.zeros((100, 100))
cross_sprite.render(canvas)
cross_sprite2.render(canvas)

print(cross_sprite.collide(cross_sprite2))
print(cross_sprite2.collide(cross_sprite))

plt.imshow(canvas)
plt.show()
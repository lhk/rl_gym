import matplotlib.pyplot as plt
import numpy as np

from np_draw.sprite import Sprite

cross_img = np.zeros((3, 3))
cross_img[1, :] = 1
cross_img[:, 1] = 1

cross_sprite = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)
cross_sprite2 = Sprite(cross_img, cross_img > 0, np.array([4, 3]), 0)

canvas = np.zeros((10, 10))
cross_sprite.render(canvas)
cross_sprite2.render(canvas)

plt.imshow(canvas)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from np_draw.sprite import Sprite
from unittest import TestCase

class Sprite_Test(TestCase):
    def test_rendering(self):
        cross_img = np.zeros((3, 3))
        cross_img[1, :] = 1
        cross_img[:, 1] = 1

        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)

        canvas = np.zeros((10, 10))
        cross_sprite.render(canvas)

        self.assertEqual(canvas[1, 1], 1)

    def test_collision_simple(self):
        cross_img = np.zeros((3, 3))
        cross_img[1, :] = 1
        cross_img[:, 1] = 1

        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)
        cross_sprite2 = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)

        canvas = np.zeros((10, 10))
        cross_sprite.render(canvas)
        cross_sprite2.render(canvas)

        self.assertTrue(cross_sprite.collide(cross_sprite2))

    def test_collision_rotated(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from np_draw.sprite import Sprite

        cross_img = np.zeros((30, 30))
        cross_img[13:16, :] = 1
        cross_img[:, 13:16] = 1

        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([30, 16]), 60)
        cross_sprite2 = Sprite(cross_img, cross_img > 0, np.array([25, 35]), 0)

        self.assertTrue(cross_sprite.collide(cross_sprite2))

        cross_sprite.set_rotation(75)

        self.assertTrue(not cross_sprite.collide(cross_sprite2))
from unittest import TestCase

import numpy as np

from np_draw.sprite import Sprite


class Sprite_Test(TestCase):
    def test_rendering(self):
        """
        tests the rendering to canvas,
        the changes need to be persistent, we don't work on a copy
        :return:
        """
        cross_img = np.zeros((3, 3))
        cross_img[1, :] = 1
        cross_img[:, 1] = 1

        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)

        # the return of render is not assigned to canvas
        # but canvas is still changed
        canvas = np.zeros((10, 10))
        cross_sprite.render(canvas)
        self.assertEqual(canvas[1, 1], 1)

    def test_collision_simple(self):
        """
        a simple test of collision between cross shaped graphics.
        :return:
        """
        cross_img = np.zeros((3, 3))
        cross_img[1, :] = 1
        cross_img[:, 1] = 1

        # these obviously collide
        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)
        cross_sprite2 = Sprite(cross_img, cross_img > 0, np.array([1, 1]), 0)
        self.assertTrue(cross_sprite.collide(cross_sprite2))

        # but for this position, the pixels don't collide
        cross_sprite3 = Sprite(cross_img, cross_img > 0, np.array([4, 3]), 0)
        self.assertFalse(cross_sprite.collide(cross_sprite3))

    def test_collision_rotated(self):
        """
        a testcase for pixel-wise collision under rotation
        :return:
        """

        cross_img = np.zeros((30, 30))
        cross_img[13:16, :] = 1
        cross_img[:, 13:16] = 1

        # with this constellation, the crosses touch
        cross_sprite = Sprite(cross_img, cross_img > 0, np.array([30, 16]), 60)
        cross_sprite2 = Sprite(cross_img, cross_img > 0, np.array([25, 35]), 0)
        self.assertTrue(cross_sprite.collide(cross_sprite2))

        # but if one of them is rotated a bit more, they no longer touch
        cross_sprite.set_rotation(75)
        self.assertFalse(cross_sprite.collide(cross_sprite2))

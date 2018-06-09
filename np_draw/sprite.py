import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt


def render(img, mask, upperleft, canvas):
    assert np.all(upperleft >= 0), "image must be withing canvas"
    assert np.all(upperleft + img.shape[:2] <= canvas.shape[:2]), "image must be within canvas"

    shape = img.shape[:2]
    canvas_slice = canvas[upperleft[0]:upperleft[0] + shape[0],
                   upperleft[1]:upperleft[1] + shape[1]]

    canvas_slice[mask] = img[mask]

    return canvas


class Sprite():
    def __init__(self, img, mask, pos, rot):
        self.img = img.copy()
        self.mask = mask.copy()

        if len(img.shape) > 2:
            colors = img.shape[2]
            self.mask = np.stack([self.mask] * colors, axis=-1)

        self.pos = pos
        self.dim = np.linalg.norm(self.img.shape[:2])
        self.set_rotation(rot)

    def set_rotation(self, new_rot):
        self.rot = new_rot
        self.img_rotated = rotate(self.img, self.rot, resize=True, order=1)
        self.mask_rotated = rotate(self.mask, self.rot, resize=True, order=1).astype(np.bool)

        self.size = np.array(self.img_rotated.shape[:2])
        self.upperleft = self.pos - self.size / 2

        self.size_int = self.size.astype(np.int)
        self.upperleft_int = self.upperleft.astype(np.int)

        self.masked_img = self.img_rotated[self.mask_rotated]

    def set_position(self, new_pos):
        self.pos = new_pos
        self.upperleft = self.pos - self.size / 2
        self.upperleft_int = self.upperleft.astype(np.int)

    def render(self, canvas):

        # is image outside of canvas ?
        if np.all(self.upperleft > canvas.shape[:2]):
            return

        # if upperleft is negative, the image is not visible,
        # cut away those parts
        cutoff = np.abs(self.upperleft_int) * (self.upperleft_int < 0)
        new_upperleft = np.clip(self.upperleft_int, 0, None)

        img_visible = self.img_rotated[cutoff[0]:, cutoff[1]:]
        mask_visible = self.mask_rotated[cutoff[0]:, cutoff[1]:]

        if img_visible.shape[0] == 0 or img_visible.shape[1] == 0:
            return

        # the image can also move out of canvas on the other side
        # we have to cut again
        visible = np.minimum(-new_upperleft + canvas.shape[:2], img_visible.shape[:2])

        img_visible = img_visible[:visible[0], :visible[1]]
        mask_visible = mask_visible[:visible[0], :visible[1]]

        upperleft = np.clip(self.upperleft_int, 0, max(canvas.shape))
        return render(img_visible, mask_visible, upperleft, canvas)

    def collide(self, other):
        # if distance is high enough, collisions are impossible
        dist_vec = self.pos - other.pos
        dist = np.linalg.norm(dist_vec)
        if dist > self.dim + other.dim:
            return False

        dist_vec = self.upperleft - other.upperleft

        # we have to check for pixel precise collision
        # render the two images and compare pixels
        offset = (dist_vec < 0).astype(np.int) * -dist_vec
        other_pos = offset.astype(np.int)
        own_pos = (dist_vec + offset).astype(np.int)

        max_x = max(own_pos[0] + self.size[0], other_pos[0] + other.size[0])
        max_y = max(own_pos[1] + self.size[1], other_pos[1] + other.size[1])

        own_canvas = np.zeros((max_x, max_y, self.img_rotated.shape[2]))
        other_canvas = np.zeros((max_x, max_y, other.img_rotated.shape[2]))

        render(self.mask_rotated, self.mask_rotated, own_pos, own_canvas)
        render(other.mask_rotated, other.mask_rotated, other_pos, other_canvas)

        intersection = (own_canvas > 0) * (other_canvas > 0)
        return np.any(intersection)

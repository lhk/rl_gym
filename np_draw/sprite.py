import numpy as np
from skimage.transform import rotate


def render(img, mask, upperleft, canvas):
    assert np.all(upperleft >= 0), "image must be withing canvas"
    assert np.all(upperleft + img.shape <= canvas.shape), "image must be within canvas"

    shape = img.shape[:2]
    canvas_slice = canvas[upperleft[0]:upperleft[0] + shape[0],
                   upperleft[1]:upperleft[1] + shape[1]]
    area = canvas_slice.shape[:2]

    mask_slice = mask[-area[0]:, -area[1]:]
    img_slice = img[-area[0]:, -area[1]:]
    canvas_slice[mask_slice] = img_slice[mask_slice]

    return canvas


class Sprite():
    def __init__(self, img, mask, pos, rot):
        self.img = img.copy()
        self.mask = mask.copy()
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

    def render(self, canvas):
        # cut away parts of the image that are outside of the canvas
        # after that, upperleft can be clipped
        cutoff = np.abs(self.upperleft_int) * (self.upperleft_int < 0)
        img_visible = self.img_rotated[cutoff[0]:, cutoff[1]:]
        img_visible = img_visible[:canvas.shape[0], :canvas.shape[1]]

        mask_visible = self.mask_rotated[cutoff[0]:, cutoff[1]:]
        mask_visible = mask_visible[:canvas.shape[0], :canvas.shape[1]]

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

        own_canvas = np.zeros((max_x, max_y))
        other_canvas = np.zeros((max_x, max_y))

        render(self.img_rotated, self.mask_rotated, own_pos, own_canvas)
        render(other.img_rotated, other.mask_rotated, other_pos, other_canvas)

        intersection = (own_canvas > 0) * (other_canvas > 0)
        return np.any(intersection)

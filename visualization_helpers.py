import matplotlib.pyplot as plt


def show_img(img):
    assert len(img.shape) in [2, 3], "x, y, [color]"
    if len(img.shape) == 3:
        assert img.shape[2] == 3, "rgb"

    plt.imshow(img)
    plt.show()

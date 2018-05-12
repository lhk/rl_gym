import numpy as np

items = 4  # 2^10 ~= 10^3

_length = 2*items - 1
_offset = items -1

# to store the errors
data = np.zeros((_length, 1))


# update or insert a new element
def update(idx, val):
    arr_idx = _offset + idx

    delta = val - data[arr_idx]
    data[arr_idx] = val

    parent_idx = (arr_idx - 1) // 2
    arr_idx = parent_idx
    data[arr_idx] -= delta

    while (parent_idx != 0):
        parent_idx = (arr_idx - 1) // 2
        arr_idx = parent_idx
        data[arr_idx] += delta

update(2, 1)
update(1, 3)
update(3, 1)
update(4, 1)

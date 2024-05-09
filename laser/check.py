import numpy as np


def load_vector(surface: np.ndarray, nodes_num: int, q: float, circle_center: list, r: float) -> np.ndarray:
    f = np.zeros((nodes_num, 1))

    x0, z0 = circle_center
    lst = []

    for j in range(len(surface)):

        x, z = surface[j, 1], surface[j, 3]
        if (x-x0)**2 + (z-z0)**2 < r**2:
            lst.append(int(surface[j, 0]))

    for i in lst:
        f[i] = q / 3

    return f

__author__ = 'aymgal'

import numpy as np


def shift_center(x, y, x0, y0):
    # shift the center of the coordinates
    return x-x0, y-y0

def rotate_axes(x, y, phi):
    # apply the rotation
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x1 = cos_phi*x + sin_phi*y
    x2 = -sin_phi*x + cos_phi*y
    return x1, x2

def shift_and_rotate(x, y, x0, y0, phi):
    xs, ys = shift_center(x, y, x0, y0)
    x1, x2 = rotate_axes(xs, ys, phi)
    return x1, x2

__author__ = 'aymgal'

import numpy as np


def square_grid(num_pix, delta_pix=1, step=1, dtype=float):
    a = np.arange(0, num_pix, step=step) * delta_pix
    mesh = np.dstack(np.meshgrid(a, a)).reshape(-1, 2).astype(dtype)
    x = mesh[:, 0]
    y = mesh[:, 1]
    return x, y

def shift_center(x, y, x0, y0):
    # shift the center of the coordinates
    return x-x0, y-y0

def rotate_axes(x, y, phi):
    # apply the rotation
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x1 = cos_phi*x + sin_phi*y
    x2 = -sin_phi*x + cos_phi*y
    return x1, x2

def apply_ellipse(x, y, e1, e2):
    # apply elliptical transformation through excentricities
    x1 = (1 - e1) * x - e2 * y
    x2 = - e2 * x + (1 + e1) * y
    det = np.sqrt((1-e1)*(1+e1) + e2**2)
    return x1/det, x2/det

def shift_and_rotate(x, y, x0, y0, phi):
    xs, ys = shift_center(x, y, x0, y0)
    x1, x2 = rotate_axes(xs, ys, phi)
    return x1, x2

def shift_and_ellipse(x, y, x0, y0, phi, q):
    xs, ys = shift_center(x, y, x0, y0)
    q_fac = (1.-q) / (1.+q)
    e1 = q_fac * np.cos(2*phi)
    e2 = q_fac * np.sin(2*phi)
    x1, x2 = apply_ellipse(xs, ys, e1, e2)
    return x1, x2

def array_to_image(one_d_array):
    n2 = np.size(one_d_array)
    n  = int(np.sqrt(n2))
    two_d_shape = (n, n)
    try:
        two_d_array = one_d_array.reshape(two_d_shape)
    except ValueError as e:
        raise ValueError("Image needs to be defined on square grid !"+
                         "\nOriginal error : {}".format(e))
    return two_d_array
    
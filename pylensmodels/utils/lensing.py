__author__ = 'aymgal'


def hessian2mag(f_xx, f_yy, f_xy, f_yx):
    det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
    mu = 1./det_A
    return mu

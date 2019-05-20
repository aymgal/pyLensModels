__author__ = 'aymgal'

import numpy as np

import pylensmodels.utils.constants as cst


def critical_lines_pixel(kappa_map, gamma_map, eps_t=1e-3, eps_r=1e-2):
    """Compute the critical lines on a pixelized grid
    Everywhere else is np.nan so it appears transparent on plt.imshow()
    """
    img_shape = kappa_map.shape
    
    # eigenvalues of the total magnification tensor M = A**-1
    lambda_t = 1. - kappa_map - gamma_map
    lambda_r = 1. - kappa_map + gamma_map

    # tangential critical line
    condition_tangential = (-eps_t < lambda_t) & (lambda_t < eps_t)  # mimics the 'lambda_t = 0' condition

    critical_line_t = np.nan*np.ones(img_shape)
    critical_line_t[np.where(condition_tangential)] = 1

    # radial critical line
    condition_radial = (-eps_r < lambda_r) & (lambda_r < eps_r)  # mimics the 'lambda_r = 0' condition

    critical_line_r = np.nan*np.ones(img_shape)
    critical_line_r[np.where(condition_radial)] = 1
    
    return critical_line_t, critical_line_r


def caustics_pixel(kappa_map, gamma_map, raytrace_func, eps_t=1e-3, eps_r=1e-2):
    """Compute the caustics on a pixelized grid.
    It simply (back)ray trace critical lines to source plane. 
    Everywhere else is np.nan so it appears transparent on plt.imshow()
    """
    crit_lines = critical_lines_pixel(kappa_map, gamma_map, 
                                      eps_t=eps_t, eps_r=eps_r)
    caustics = []
    for crit_line in crit_lines:
        crit_line[np.where(np.isnan(crit_line))] = 0
        caustic = raytrace_func(crit_line)
        caustic[caustic > 0]  = 1
        caustic[caustic == 0] = np.nan
        caustics.append(caustic)
    return caustics


def critical_surface_density(Ds, Dd, Dds):
    return cst.c**2 / (4.*np.pi*cst.G) * Ds / (Dd*Dds)


def hessian2mag(f_xx, f_yy, f_xy, f_yx):
    det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
    mu = 1./det_A
    return mu

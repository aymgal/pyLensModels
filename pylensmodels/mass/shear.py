__author__ = 'aymgal'

import numpy as np
from fastell4py import fastell4py

from pylensmodels.mass.base import BaseMassModel
import pylensmodels.mass.conversions as conv
import pylensmodels.utils.coordinates as coord


_defaults = {
    'x0': 0.,
    'y0': 0.,
    'phi': 2.,
    'gamma_ext': 10.
}

class ExternalShear_glee(BaseMassModel):

    def __init__(self, kwargs_parameters):
        super(ExternalShear_glee, self).__init__(kwargs_parameters)
        self._extract_values(kwargs_parameters)

    def potential(self, x, y):
        # apply the center shift
        xs, ys = coord.shift_center(x, y, self.x0, self.y0)
        # compute the potential Psi
        psi = 0.5 * self.gamma_ext * ( np.cos(2.*self.phi) * (xs**2-ys**2) + 2.*np.sin(2.*self.phi) * xs*ys )
        return psi

    def derivative(self, x, y):
        assert x.shape == y.shape, "External shear requires same axis dimensions"
        xs, ys = coord.shift_center(x, y, self.x0, self.y0)
        e1 = self.gamma_ext*np.cos(2.*self.phi)
        e2 = self.gamma_ext*np.sin(2.*self.phi)
        f_x = xs * e1 + ys * e2
        f_y = xs * e2 - ys * e1
        return f_x, f_y

    def hessian(self, x, y):
        assert x.shape == y.shape, "External shear requires same axis dimensions"
        e1 = self.gamma_ext*np.cos(2.*self.phi)
        e2 = self.gamma_ext*np.sin(2.*self.phi)
        kappa = 0.
        gamma1 = e1
        gamma2 = e2
        f_xx = np.ones_like(x) * (kappa + gamma1)
        f_yy = np.ones_like(x) * (kappa - gamma1)
        f_xy = np.ones_like(x) * gamma2
        f_yx = np.ones_like(x) * f_xy
        return f_xx, f_yy, f_xy, f_yx

    def _extract_values(self, kw_params):
        self.gamma_ext = self._get_value('gamma_ext', kw_params, _defaults)
        self.phi = self._get_value('phi', kw_params, _defaults)
        self.x0 = self._get_value('x0', kw_params, _defaults)
        self.y0 = self._get_value('y0', kw_params, _defaults)

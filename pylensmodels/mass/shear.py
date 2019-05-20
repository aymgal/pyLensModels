__author__ = 'aymgal'

import numpy as np
from fastell4py import fastell4py

from pylensmodels.mass.base_mass import BaseMassModel
import pylensmodels.mass.conversions as conv
import pylensmodels.utils.coordinates as coord


_defaults = {
    'x0': 0.,
    'y0': 0.,
    'phi_ext': 2.,
    'gamma_ext': 10.
}

class ExternalShear_glee(BaseMassModel):

    def __init__(self, kwargs_parameters, **base_class_kwargs):
        super().__init__(kwargs_parameters, **base_class_kwargs)
        self._extract_values(kwargs_parameters)

    def function(self, x, y):
        # apply the center shift
        xs, ys = coord.shift_center(x, y, self.x0, self.y0)

        # compute the potential Psi
        gamma_cos2phi = self.gamma_ext * np.cos(2.*self.phi_ext)
        gamma_sin2phi = self.gamma_ext * np.sin(2.*self.phi_ext)
        psi = 0.5 * gamma_cos2phi * (xs**2 - ys**2) + gamma_sin2phi * xs*ys

        #other formulation (equivalent) with polar coordinates
        # r = np.sqrt(xs**2 + ys**2)
        # phi = np.arctan2(ys, xs)
        # psi = 0.5 * self.gamma_ext * r**2 * np.cos(2. * ( phi - self.phi_ext ))

        return psi

    def derivative(self, x, y):
        assert x.shape == y.shape, "External shear requires same axis dimensions"
        xs, ys = coord.shift_center(x, y, self.x0, self.y0)
        gamma_cos2phi = self.gamma_ext * np.cos(2.*self.phi_ext)
        gamma_sin2phi = self.gamma_ext * np.sin(2.*self.phi_ext)
        f_x = xs * gamma_cos2phi + ys * gamma_sin2phi
        f_y = xs * gamma_sin2phi - ys * gamma_cos2phi
        return f_x, f_y

    def hessian(self, x, y):
        assert x.shape == y.shape, "External shear requires same axis dimensions"
        gamma_cos2phi = self.gamma_ext * np.cos(2.*self.phi_ext)
        gamma_sin2phi = self.gamma_ext * np.sin(2.*self.phi_ext)
        kappa  = 0.
        gamma1 = gamma_cos2phi
        gamma2 = gamma_sin2phi
        f_xx = np.ones_like(x) * (kappa + gamma1)
        f_yy = np.ones_like(x) * (kappa - gamma1)
        f_xy = np.ones_like(x) * gamma2
        f_yx = np.ones_like(x) * f_xy
        return f_xx, f_yy, f_xy, f_yx

    def _extract_values(self, kw_params):
        self.gamma_ext = self._get_value('gamma_ext', kw_params, _defaults)
        self.phi_ext = self._get_value('phi_ext', kw_params, _defaults)
        self.x0 = self._get_value('x0', kw_params, _defaults)
        self.y0 = self._get_value('y0', kw_params, _defaults)

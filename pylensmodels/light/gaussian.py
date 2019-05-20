__author__ = 'aymgal'

import numpy as np
import fastell4py.fastell4py as fl

from pylensmodels.light.base_light import BaseLightModel
import pylensmodels.utils.coordinates as coord


_defaults = {
    'x0': 0.,
    'y0': 0.,
    'sigma_x': 1.,
    'sigma_y': 1.,
    'amp': 1.,

    # only for elliptical Gaussian
    'sigma': 1.,
    'phi': 0.,
    'q': 1.
}

class Gaussian(BaseLightModel):

    def __init__(self, kwargs_parameters, **base_class_kwargs):
        super().__init__(kwargs_parameters, **base_class_kwargs)
        self._extract_values(kwargs_parameters)

    def function(self, x, y):
        xs, ys = coord.shift_center(x, y, self.x0, self.y0)
        C = self.amp / (2. * np.pi * self.sigma_x * self.sigma_y)
        R2 = (xs/self.sigma_x)**2 + (ys/self.sigma_y)**2
        gaussian_1d = C * np.exp(-R2 / 2.)
        return coord.array_to_image(gaussian_1d)

    def _extract_values(self, kw_params):
        self.x0 = self._get_value('x0', kw_params, _defaults)
        self.y0 = self._get_value('y0', kw_params, _defaults)
        self.sigma_x = self._get_value('sigma_x', kw_params, _defaults)
        self.sigma_y = self._get_value('sigma_y', kw_params, _defaults)
        self.amp = self._get_value('amp', kw_params, _defaults)


class GaussianElliptical(BaseLightModel):

    def __init__(self, kwargs_parameters, **base_class_kwargs):
        super().__init__(kwargs_parameters, **base_class_kwargs)
        self._extract_values(kwargs_parameters)

    def function(self, x, y):
        x1, x2 = coord.shift_and_ellipse(x, y, self.x0, self.y0, self.phi, self.q)
        C = self.amp / (2. * np.pi * self.sigma**2)
        R2 = (x1/self.sigma)**2 + (x2/self.sigma)**2
        gaussian_1d = C * np.exp(-R2 / 2.)
        return coord.array_to_image(gaussian_1d)

    def _extract_values(self, kw_params):
        self.x0 = self._get_value('x0', kw_params, _defaults)
        self.y0 = self._get_value('y0', kw_params, _defaults)
        self.sigma = self._get_value('sigma', kw_params, _defaults)
        self.phi = self._get_value('phi', kw_params, _defaults)
        self.q = self._get_value('q', kw_params, _defaults)
        self.amp = self._get_value('amp', kw_params, _defaults)

__author__ = 'aymgal'

import numpy as  np

from pylensmodels.utils import lensing
import pylensmodels.utils.coordinates as coord


def _not_implemented_error(method_name):
    message = "{} should be implemented in subclass".format(method_name)
    raise NotImplementedError(message)


class BaseMassModel(object):

    def __init__(self, kwargs_parameters):
        self.init_parameters = kwargs_parameters

    def potential(self, x, y):
        return _not_implemented_error(self.potential.__name__)

    def derivative(self, x, y):
        return _not_implemented_error(self.deflection.__name__)

    def hessian(self, x, y):
        return _not_implemented_error(self.hessian.__name__)

    def derivative_numdiff(self, x, y, diff=1e-7, method='2-points'):
        """using numerical differentiation from 1st order derivatives"""
        f = self.potential(x, y)
        f_dx = self.potential(x + diff, y)
        f_dy = self.potential(x, y + diff)

        # differentiation
        if method == '2-points':
            f_x = (f_dx - f) / diff
            f_y = (f_dy - f) / diff
        else:
            raise NotImplementedError("Only the 2-points method is currently implemented")
        return f_x, f_y

    def hessian_numdiff(self, x, y, diff=1e-7, method='2-points'):
        """using numerical differentiation from 1st order derivatives"""
        f_x, f_y = self.derivative(x, y)
        f_x_dx, f_y_dx = self.derivative(x + diff, y)
        f_x_dy, f_y_dy = self.derivative(x, y + diff)

        # differentiation
        if method == '2-points':
            f_xx = (f_x_dx - f_x) / diff
            f_yy = (f_y_dy - f_y) / diff
            f_xy = (f_x_dy - f_x) / diff
            f_yx = (f_y_dx - f_y) / diff
        else:
            raise NotImplementedError("Only the 2-points method is currently implemented")
        return f_xx, f_yy, f_xy, f_yx

    def deflection(self, x, y, numdiff_kwargs=None):
        """return deflection angles"""
        if numdiff_kwargs is None:
            f_x, f_y = self.derivative(x, y)
        else:
            f_x, f_y = self.derivative_numdiff(x, y, **numdiff_kwargs)
        alpha1 = f_x
        alpha2 = f_y
        n2 = np.size(x)
        alpha1 = coord.array_to_image(alpha1, n2)
        alpha2 = coord.array_to_image(alpha2, n2)
        return alpha1, alpha2

    def convergence(self, x, y, numdiff_kwargs=None):
        """return convergence map"""
        if numdiff_kwargs is None:
            f_xx, f_yy, _, _ = self.hessian(x, y)
        else:
            f_xx, f_yy, _, _ = self.hessian_numdiff(x, y, **numdiff_kwargs)
        kappa = 0.5 * (f_xx + f_yy)  # convergence
        n2 = np.size(x)
        kappa = coord.array_to_image(kappa, n2)
        return kappa

    def shear(self, x, y, numdiff_kwargs=None):
        """return shear map"""
        if numdiff_kwargs is None:
            f_xx, f_yy, f_xy, f_yx = self.hessian(x, y)
        else:
            f_xx, f_yy, f_xy, f_yx = self.hessian_numdiff(x, y, **numdiff_kwargs)
        gamma1 = 0.5 * (f_xx - f_yy) # shear, 1st component
        gamma2 = f_xy # shear, 2nd component
        n2 = np.size(x)
        gamma1 = coord.array_to_image(gamma1, n2)
        gamma2 = coord.array_to_image(gamma2, n2)
        return gamma1, gamma2

    def magnification(self, x, y, numdiff_kwargs=None):
        """return magnification map"""
        if numdiff_kwargs is None:
            f_xx, f_yy, f_xy, f_yx = self.hessian(x, y)
        else:
            f_xx, f_yy, f_xy, f_yx = self.hessian_numdiff(x, y, **numdiff_kwargs)
        mu = lensing.hessian2mag(f_xx, f_yy, f_xy, f_yx)
        n2 = np.size(x)
        mu = coord.array_to_image(mu, n2)
        return mu

    def _get_value(self, name, kw_params, kw_defaults):
        return kw_params.get(name, kw_defaults[name])

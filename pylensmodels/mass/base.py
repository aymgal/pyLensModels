__author__ = 'aymgal'

from pylensmodels.utils import lensing


def _not_implemented_error(method_name):
    message = "{} should be implemented in subclass".format(method_name)
    raise NotImplementedError(message)


class BaseMassModel(object):

    def __init__(self, kwargs_parameters):
        self.init_parameters = kwargs_parameters

    def potential(self, x, y):
        return _not_implemented_error(self.potential.__name__)

    def deflection(self, x, y):
        return _not_implemented_error(self.deflection.__name__)

    def convergence(self, x, y):
        return _not_implemented_error(self.convergence.__name__)

    def shear(self, x, y):
        return _not_implemented_error(self.shear.__name__)

    def magnification(self, x, y):
        """return magnification map"""
        f_xx, f_yy, f_xy, f_yx = self.hessian(x, y)
        return lensing.hessian2mag(f_xx, f_yy, f_xy, f_yx)

    def derivative(self, x, y):
        return _not_implemented_error(self.deflection.__name__)

    def hessian(self, x, y):
        return _not_implemented_error(self.hessian.__name__)

    def _get_value(self, name, kw_params, kw_defaults):
        return kw_params.get(name, kw_defaults[name])

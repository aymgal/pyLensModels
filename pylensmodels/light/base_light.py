__author__ = 'aymgal'

import numpy as  np

from pylensmodels.utils import lensing
import pylensmodels.utils.coordinates as coord


def _not_implemented_error(method_name):
    message = "{} should be implemented in subclass".format(method_name)
    raise NotImplementedError(message)


class BaseLightModel(object):

    def __init__(self, kwargs_parameters):
        self.init_parameters = kwargs_parameters

    def function(self, x, y):
        """return potential (as 1D array)"""
        return _not_implemented_error(self.potential.__name__)

    def derivative(self, x, y):
        """return 1st derivatives (as 1D array)"""
        return _not_implemented_error(self.deflection.__name__)

    def hessian(self, x, y):
        """return 2nd derivatives (as 1D array)"""
        return _not_implemented_error(self.hessian.__name__)

    def _get_value(self, name, kw_params, kw_defaults):
        return kw_params.get(name, kw_defaults[name])
        
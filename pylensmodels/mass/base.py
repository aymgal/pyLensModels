__author__ = 'aymgal'


def _not_implemented_errpor(method_name):
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
        return _not_implemented_error(self.magnification.__name__)

    def derivative(self, x, y):
        return _not_implemented_error(self.deflection.__name__)

    def hessian(self, x, y):
        return _not_implemented_error(self.hessian.__name__)

__author__ = 'aymgal'

import numpy as np
from fastell4py import fastell4py

from pylensmodels.mass.base import BaseMassModel
from pylensmodels.utils import lensing

class MultiMass(object):

    def __init__(self, model_list):
        self.models = []
        for model in model_list:
            if isinstance(model, BaseMassModel):
                self.models.append(model)
            else:
                print("WARNING : model {} is not an instance of the base class (model ignored)")

    def potential(self, x, y):
        pot = 0.
        for model in self.models:
            pot += model.potential(x, y)
        return pot

    def derivative(self, x, y):
        f_x, f_y = 0., 0.
        for model in self.models:
            f_x_, f_y_ = model.derivative(x, y)
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y):
        f_xx, f_yy, f_xy, f_yx = 0., 0., 0., 0.
        for model in self.models:
            f_xx_, f_yy_, f_xy_, f_yx_ = model.hessian(x, y)
            f_xx += f_xx_
            f_yy += f_yy_
            f_xy += f_xy_
            f_yx += f_yx_
        return f_xx, f_yy, f_xy, f_yx

    def magnification(self, x, y):
        """return magnification map"""
        f_xx, f_yy, f_xy, f_yx = self.hessian(x, y)
        return lensing.hessian2mag(f_xx, f_yy, f_xy, f_yx)

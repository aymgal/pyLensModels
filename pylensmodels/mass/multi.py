__author__ = 'aymgal'

import numpy as np
from fastell4py import fastell4py

from pylensmodels.mass.base import BaseMassModel
from pylensmodels.utils import lensing


class MultiMass(BaseMassModel):
    """The distance ratio scaling (Dds/Ds) is assumed to be performed by 
    individual model class methods 'potential', 'derivative', 'hessian'.
    """

    def __init__(self, model_list, **base_class_kwargs):
        self.model_list = []
        kwargs_parameters_list = []
        for model in model_list:
            if isinstance(model, BaseMassModel):
                self.model_list.append(model)
                kwargs_parameters_list.append(model.init_parameters)
            else:
                print("WARNING : model {} is not an instance of the base class (model ignored)")
        
        # we pass to the mother class a lot of kwargs
        # (ATTENTION, to be changed in future)
        super().__init__(kwargs_parameters_list, **base_class_kwargs)

    def function(self, x, y):
        pot = 0.
        for model in self.model_list:
            pot += model.potential(x, y)
        return pot

    def derivative(self, x, y):
        f_x, f_y = 0., 0.
        for model in self.model_list:
            f_x_, f_y_ = model.derivative(x, y)
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y):
        f_xx, f_yy, f_xy, f_yx = 0., 0., 0., 0.
        for model in self.model_list:
            f_xx_, f_yy_, f_xy_, f_yx_ = model.hessian(x, y)
            f_xx += f_xx_
            f_yy += f_yy_
            f_xy += f_xy_
            f_yx += f_yx_
        return f_xx, f_yy, f_xy, f_yx

    def derivative_numdiff(self, x, y, diff=1e-7, method='2-points'):
        f_x, f_y = 0., 0.
        for model in self.model_list:
            f_x_, f_y_ = model.derivative_numdiff(x, y, diff=diff, method=method)
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian_numdiff(self, x, y, diff=1e-7, method='2-points', recursive=False):
        f_xx, f_yy, f_xy, f_yx = 0., 0., 0., 0.
        for model in self.model_list:
            f_xx_, f_yy_, f_xy_, f_yx_ = \
                model.hessian_numdiff(x, y, diff=diff, method=method, 
                                      recursive=recursive)
            f_xx += f_xx_
            f_yy += f_yy_
            f_xy += f_xy_
            f_yx += f_yx_
        return f_xx, f_yy, f_xy, f_yx

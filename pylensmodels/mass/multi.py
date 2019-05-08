__author__ = 'aymgal'

import numpy as np
from fastell4py import fastell4py

from pylensmodels.mass.base import BaseMassModel
from pylensmodels.utils import lensing


class MultiMass(BaseMassModel):

    def __init__(self, model_list, Dds_Ds=None):
        self.model_list = model_list
        self.models = []
        for model in model_list:
            if isinstance(model, BaseMassModel):
                self.models.append(model)
            else:
                print("WARNING : model {} is not an instance of the base class (model ignored)")
        self._Dds_Ds = Dds_Ds

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

        if self._Dds_Ds is not None:
            # scale second derivatives by the distance ratio Dds/Ds
            f_xx *= self._Dds_Ds
            f_yy *= self._Dds_Ds
            f_xy *= self._Dds_Ds
            f_yx *= self._Dds_Ds

        return f_xx, f_yy, f_xy, f_yx

    # def deflection(self, x, y):
    #     alpha1, alpha2 = 0., 0.
    #     for model in self.models:
    #         alpha1_, alpha2_ = model.deflection(x, y)
    #         alpha1 += alpha1_
    #         alpha2 += alpha2_
    #     return alpha1, alpha2

    # def shear(self, x, y):
    #     gamma1, gamma2 = 0., 0.
    #     for model in self.models:
    #         gamma1_, gamma2_ = model.shear(x, y)
    #         gamma1 += gamma1_
    #         gamma2 += gamma2_
    #     return gamma1, gamma2

    # def convergence(self, x, y):
    #     kappa = 0.
    #     for model in self.models:
    #         kappa += model.convergence(x, y)
    #     return kappa

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

    def magnification_numdiff(self, x, y, diff=1e-7):
        """return magnification map"""
        f_xx, f_yy, f_xy, f_yx = self.hessian_numdiff(x, y, diff=diff)
        return lensing.hessian2mag(f_xx, f_yy, f_xy, f_yx)

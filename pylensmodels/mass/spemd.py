__author__ = 'aymgal'

import numpy as np
import fastell4py.fastell4py as fl

from pylensmodels.mass.base import BaseMassModel
import pylensmodels.mass.conversions as conv
import pylensmodels.utils.coordinates as coord


_defaults = {
    'x0': 0.,
    'y0': 0.,
    'gamma': 2.,
    'theta_E': 10.,
    'q': 1.,
    'phi': 0.,
    'r_core': 0.01
}

class SPEMD_glee(BaseMassModel):

    def __init__(self, kwargs_parameters, rotation_fix=True):
        super(SPEMD_glee, self).__init__(kwargs_parameters)
        self._rotation_fix = rotation_fix
        self._extract_values(kwargs_parameters)

    def potential(self, x, y):
        """return the SPEMD 2D potential
        'phi' is the position angle, 'theta_E' is the Einstein radius
        """
        x1, x2 = coord.shift_and_rotate(x, y, self.x0, self.y0, self.phi)
        
        # call Fastell's routine
        psi = fl.ellipphi(x1, x2, self.q_fastell, self.gamma, 
                                  arat=self.arat, s2=self.s2)
        return psi

    def derivative(self, x, y):
        """return 1st order derivatives"""
        x1, x2 = coord.shift_and_rotate(x, y, self.x0, self.y0, self.phi)
        
        # call Fastell's routine
        f_x_, f_y_ = fl.fastelldefl(x1, x2, self.q_fastell, self.gamma, 
                                            arat=self.q, s2=self.s2)
        
        if self._rotation_fix:
            cos_phi, sin_phi = np.cos(self.phi), np.sin(self.phi)
            f_x = cos_phi*f_x_ - sin_phi*f_y_
            f_y = sin_phi*f_x_ + cos_phi*f_y_
        else:
            f_x = f_x_
            f_y = f_y_
        return f_x, f_x

    def hessian(self, x, y):
        """return 2nd order derivatives"""
        x1, x2 = coord.shift_and_rotate(x, y, self.x0, self.y0, self.phi)
        
        # call Fastell's routine
        _, _, f_xx_, f_yy_, f_xy_ = fl.fastellmag(x1, x2, self.q_fastell, 
                                                  self.gamma, arat=self.q, 
                                                  s2=self.s2)
        
        if self._rotation_fix:
            kappa = (f_xx_ + f_yy_) / 2.
            gamma1_ = (f_xx_ - f_yy_) / 2.
            gamma2_ = f_xy_

            cos_2phi = np.cos(2.*self.phi)
            sin_2phi = np.sin(2.*self.phi)
            gamma1 = cos_2phi*gamma1_ - sin_2phi*gamma2_
            gamma2 = sin_2phi*gamma1_ + cos_2phi*gamma2_

            f_xx = kappa + gamma1
            f_yy = kappa - gamma1
            f_xy = gamma2
        else:
            f_xx = f_xx_
            f_yy = f_yy_
            f_xy = f_xy_

        f_yx = f_xy
        return f_xx, f_yy, f_xy, f_yx

    def deflection(self, x, y):
        """return deflection angles"""
        f_x, f_y = self.derivative(x, y)
        alpha1 = f_x
        alpha2 = f_y
        return alpha1, alpha2

    def convergence(self, x, y):
        """return convergence map"""
        f_xx, f_yy, _, _ = self.hessian(x, y)    
        kappa = (f_xx + f_yy) / 2.  # convergence
        return kappa

    def shear(self, x, y):
        """return shear map"""
        f_xx, f_yy, f_xy, f_yx = self.hessian(x, y)
        gamma1 = 0.5 * (f_xx - f_yy) # shear, 1st component
        gamma2 = f_xy # shear, 2nd component
        return gamma1, gamma2

    def _extract_values(self, kw_params):
        self.theta_E = self._get_value('theta_E', kw_params, _defaults)
        self.gamma = self._get_value('gamma', kw_params, _defaults)
        self.x0 = self._get_value('x0', kw_params, _defaults)
        self.y0 = self._get_value('y0', kw_params, _defaults)
        self.q = self._get_value('q', kw_params, _defaults)
        self.phi = self._get_value('phi', kw_params, _defaults)
        self.r_core = self._get_value('r_core', kw_params, _defaults)
        self._add_conversions()

    def _add_conversions(self):
        self.q_fastell, self.arat, self.s2 \
            = conv.glee2fastell(self.theta_E, self.q, self.r_core, self.gamma)

__author__ = 'aymgal'


def glee2fastell(theta_E, q, r_core, gamma):
    # convert to Fastell's conventions
    s = 2*r_core / (1+q)
    s2 = s**2
    theta_E2 = theta_E**2
    one_min_gam = 1 - gamma
    q_fastell = 2./(1+q) * one_min_gam * theta_E2 / ( (theta_E2 + s2)**one_min_gam - s2**one_min_gam )
    arat = q
    return q_fastell, arat, s2
    
__author__ = 'aymgal'

from astropy.cosmology import LambdaCDM


def distance_lcdm(z1, z2, omega_m, omega_l, H0):
    cosmo = LambdaCDM(H0, omega_m, omega_l)
    a = 1. / (1. + z2)
    ctd1 = cosmo.comoving_transverse_distance(z1)
    ctd2 = cosmo.comoving_transverse_distance(z2)
    D12 = (ctd2 - ctd1) * a
    return D12.value

"""
Common functions for hmf_bias objects
"""
def dn_dm(self, z, M):
    r"""
    Derivative of the number density.

    Computes the derivative of the number density
    at the requested redshift and mass points.

    Parameters
    ----------
    self: HaloStatistics
        halo_statistics object.
    z: numpy.ndarray
        Redshift points.
    M: numpy.ndarray
        Mass points in h^{-1} Msun.

    Returns
    -------
    dn_dm: numpy.ndarray
        dn_dm[i,j], where i is the redshift axis and j the mass axis.
        Units: h^4 Mpc^{-3} Ms^{-1}.
    """
    dlnsigmadlnR = self.dlns_dlnR(z, M)
    rho_mean_0 = self._Omega_m(0) * dc.rho_crit(self.background, 0.0)
    rho_mean_0 /= self.background.h**2.0

    return rho_mean_0 / M**2.0 * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)

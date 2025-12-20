import numpy as np

from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.halo_statistics import HaloStatistics


class HMFBiasAuxiliary:
    def __init__(
        self,
        halo_statistics: HaloStatistics,
    ):
        r"""Auxiliary class computing quantities used in halo mass function and halo bias models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        halo_statistics : HaloStatistics
            An object from the `HaloStatistics` class.
        """
        self.halo_statistics = halo_statistics

    def dn_dm_fsigmanu(self, z, M, fsigmanu):
        r"""Derivative of the number density with pre-computed
        halo mass function.

        Computes the derivative of the number density
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.
        fsigmanu: numpy.ndarray
            Multiplicity function.

        Returns
        -------
        dn_dm: numpy.ndarray
            dn_dm[i,j], where i is the redshift axis and j the mass axis.
            Units: h^4 Mpc^{-3} Ms^{-1}.
        """
        rho_mean_0 = self.halo_statistics._Omega_m(0) * derived_cosmology.rho_crit(
            self.halo_statistics.background, 0.0
        )
        rho_mean_0 /= self.halo_statistics.background.h**2.0

        return -rho_mean_0 / M**2.0 * fsigmanu * self.halo_statistics.dlns_dlnM(z, M)

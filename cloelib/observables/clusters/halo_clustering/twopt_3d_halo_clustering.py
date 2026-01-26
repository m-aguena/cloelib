# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn

from cloelib.auxiliary import units
from cloelib.cosmology.cosmology import Background
from cloelib.observables.clusters.auxiliary import (
    photoz_rsd_correction,
)
from cloelib.observables.clusters.halo_clustering.halo_clustering_core import (
    HaloClusteringCore,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class TwoPoint3DHaloClustering:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        background_fid: Background,
        nonu: bool = False,
    ):

        self.core = HaloClusteringCore(matter_statistics, background_fid, nonu)

    def power_spectrum_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        """Computes Pk with RSD correction.

        Parameters
        ----------
        z: np.ndarray
           Redshift
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}
        pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3
           Shape (z.size, k.size)
        z_obs_scatter: float, numpy.ndarray
            Observed redshift scatter. If array, first dimension must be z.
        b_eff: np.ndarray
            Effective bias, must have same shape as z_obs_scatter.

        Returns
        -------
        pk_halo : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation),
            Shape (z.size, k.size, other dimensions of z_obs_scatter)
        """
        # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
        # rsd corrections (z, k, ...)
        photoz_corr0, photoz_corr1, photoz_corr2 = photoz_rsd_correction(
            self.core.matter_statistics.background, z, k, z_obs_scatter, self.core.nonu
        )

        # dark matter power spectrum (z, k)
        pk = self.core.matter_statistics.matter_power_spectrum(z, k)

        # check if z_obs_scatter has more dimensions
        ndim_z_obs_scatter = len(np.array(z_obs_scatter).shape)
        if ndim_z_obs_scatter > 1:
            # if it does, add them to pk
            extra_axes = tuple(range(2, ndim_z_obs_scatter + 1))
            pk = np.expand_dims(pk, axis=extra_axes)

        b_eff_reshaped = b_eff[:, np.newaxis]  # add k axis in position 1

        # corrected power specrum (z, k, ...)
        pk_halo = (
            b_eff_reshaped**2 * photoz_corr0
            + b_eff_reshaped * photoz_corr1
            + photoz_corr2
        ) * pk

        return pk_halo

    # IR resummation of the bao wiggles in the Pk
    # not in use currently
    def Pk_IR_func(self, k: np.array, Pk: np.ndarray) -> np.ndarray:
        """
        Infrared resummation (first order approx) to correct non-linear damping of bao wiggles

        Parameters
        ----------
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}
        Pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3

        Returns
        -------
        Pk_IR: np.ndarray
           Matter power spectrum with corrected bao wiggles in (Mpc/h)^3

        """

        ns = self.core.matter_statistics.background.ns
        h = self.core.matter_statistics.background.h
        Obh2 = self.core.matter_statistics.background.Omega_b(0.0) * h**2
        Omh2 = self.core._Omega_m(0.0) * h**2
        Tcmb = 2.73

        k *= h  #  1/Mpc
        s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1.0 + 10.0 * (Obh2) ** 0.75)
        Gamma = Omh2 / h
        AG = (
            1.0
            - 0.328 * np.log(431.0 * Omh2) * Obh2 / Omh2
            + 0.38 * np.log(22.3 * Omh2) * (Obh2 / Omh2) ** 2
        )
        Gamma = Gamma * (AG + (1.0 - AG) / (1.0 + (0.43 * k * s) ** 4))
        Theta = Tcmb / 2.7
        q = k * Theta**2 / Gamma / h
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T0 = L0 / (L0 + C0 * q * q)
        T0 /= T0[0]
        P_EH = k**ns * T0**2
        P_EH = (P_EH[:, None] * (Pk[:, 0] / P_EH[0])).T
        k /= h  # h/Mpc again

        lamb = 0.25
        kS = 0.2
        lOsc = 102.707
        qlog = np.log10(k)
        dqlog = qlog[1] - qlog[0]

        # Gaussian filtering for Pnw and Wiggle-smooth split
        Pnw = (
            P_EH
            * np.array(
                [
                    dqlog
                    / np.sqrt(2.0 * np.pi * lamb**2)
                    * (
                        np.sum(
                            np.e
                            ** (
                                -0.5
                                * (klog - qlog[(abs(qlog - klog) < 4.0 * lamb)]) ** 2
                                / lamb**2
                            )
                            * Pk[:, (abs(qlog - klog) < 4.0 * lamb)]
                            / P_EH[:, ((abs(qlog - klog) < 4.0 * lamb))],
                            axis=1,
                        )
                    )
                    for klog in qlog
                ]
            ).T
        )

        Pw = Pk - Pnw

        # Sigma2 as integral up to 0.2
        icut = k <= kS
        kcut = k[icut]
        Pnwcut = Pnw[:, icut]
        kosc = 1.0 / lOsc
        norm = 1.0 / (6.0 * np.pi**2)
        Sigma2 = norm * simps(
            Pnwcut
            * (1.0 - spherical_jn(0, kcut / kosc) + 2.0 * spherical_jn(2, kcut / kosc)),
            x=kcut,
        )

        # comput final power spectrum
        Pk_IR = Pnw + np.e ** (-(k**2) * Sigma2[:, None]) * Pw

        return Pk_IR

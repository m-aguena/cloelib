# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.hmf_bias import HMFBias
from cloelib.observables.clusters.selection_function import SelectionFunction

# import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""


class ClusterCounts:
    def __init__(
        self,
        hmfbias: HMFBias,
        selectionfunction: SelectionFunction,
        covariance: HaloCovariance,
        photoz_rsd_correction,
        integ_k_arr: np.ndarray,
        integ_mass_arr: np.ndarray,
        integ_lambda_arr: np.ndarray,
        integ_ztrue_arr: np.ndarray,
        area: float = 10313,
    ):
        """
        Initializes the cluster counts

        Parameters:
        - ....

        """
        # observable objects
        self.hmfbias = hmfbias
        self.halo_statistics = self.hmfbias.halo_statistics
        self.selectionfunction = selectionfunction
        self.covariance = covariance

        # ---------------------------------------------------------------
        # Note: This is a patch as this function is currently implemented
        # in HaloClustering, should it be moved to SelectionFunction?
        # ---------------------------------------------------------------
        self.photoz_rsd_correction = photoz_rsd_correction

        # internal values
        self.area = area

        ################### QUANTITIES FOR INTEGRATION ###################

        # integration variables
        self.integ_k_arr = integ_k_arr  # k array
        self.integ_mass_arr = integ_mass_arr  # mass array in Msun h^-1
        self.integ_lambda_arr = integ_lambda_arr  # true richness array
        self.integ_ztrue_arr = integ_ztrue_arr  # true redshift array

        # hardcoded quantities
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        # P(ltrM,ztr), this quantity is also used by cluster clustering
        self.Pltrue_M_z = self.selectionfunction.P_lnlbd(
            self.integ_ztrue_arr, self.integ_mass_arr, self.integ_lambda_arr
        )

        # volume element at the center of observed redshift bins
        self.dvdzdomega_z1z2 = derived_cosmology.dV_dzdO(
            self.hmfbias.halo_statistics.perturbations.background,
            self.integ_ztrue_arr,
            hubble_units=True,
        )

        # hmf at the center of observed redshift bins
        self.dndm_z = self.hmfbias.dn_dm(self.integ_ztrue_arr, self.integ_mass_arr)
        self.bias_z = self.hmfbias.bias(
            self.integ_ztrue_arr, self.integ_mass_arr
        )  # only work for virial overdensity

    def _compute_Plob_M_z_bin(self, lambda_min, lambda_max, integral_n_steps=31):
        """compute Plob_M_z_bin.
        Compute the probability of the observed richness
        given true mass and redshift P(lob|M,ztr) for a richness bin.

        Parameters
        ----------
        lambda_bin : int
            integer corresponding to richness bin index
        Pltrue_M_z : numpy.ndarray
            probability of true richness given true mass and redshift  P(ltr|M,ztr)

        Returns
        -------
        numpy.ndarray
            P(lob|M,ztr)
        """
        # returns # P(lob|M,ztr) for a richness bin
        l_tab = np.geomspace(lambda_min, lambda_max, integral_n_steps)
        # P(lob|ltr,ztr)
        Plob_l_z = simps(
            self.selectionfunction.P_lbdobs_lbd(
                self.integ_ztrue_arr, self.integ_lambda_arr, l_tab
            ),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF':
        #       Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.integ_ztrue_arr, self.integ_lambda_arr).T

        # P(lob|M,ztr)
        Plob_M_z = simps(
            self.Pltrue_M_z[:, :, :] * Plob_l_z[:, np.newaxis, :],
            x=self.integ_lambda_arr,
            axis=-1,
        )
        return Plob_M_z

    def _compute_volume_bin(self, z_min, z_max, lambda_min):
        """compute volume bin.
        Computes volume in a given richness redshift bin for cluster counts.

        Parameters
        ----------
        z_bin: int
            integer corresponding to redshift bin index
        lambda_bin: int
            integer corresponding to richness bin index

        Returns
        -------
        dV_dzob_bin: numpy.ndarray
            observed volume element dV/dz_ob
        """
        # computes volume in richness redshift bin for cluster counts

        z_tab = np.linspace(z_min, z_max, self.z_tab_sig)
        # P(zob|ztr)
        Pzob_z = simps(
            self.selectionfunction.P_zobs_z(z_tab, lambda_min, self.integ_ztrue_arr),
            x=z_tab,
            axis=0,
        )
        # observed volume element dV/dz_ob
        dV_dzob_bin = (
            self.dvdzdomega_z1z2 * Pzob_z * (self.area) * (np.pi**2.0 / 180.0**2.0)
        )
        # N(lob,zob)
        return dV_dzob_bin

    def _compute_counts_bin(self, dV_dzob_bin, n_lbdobs_z):
        """compute counts bin.
        Compute cluster counts in a single richness and redshift bin
        Performs integral over z_true of the the volume*n_lbdobs_z

        Parameters
        ----------
        dV_dzob_bin: numpy.ndarray
            volume element of bin
        n_lbdobs_z: numpy.ndarray
            density of clusters with observed richness and true redshift

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        # computes counts in a richness redshift bin
        return simps(n_lbdobs_z * dV_dzob_bin, x=self.integ_ztrue_arr, axis=0)

    def compute_binned_properties(self, z_obs_bins, lambda_obs_bins):
        """compute counts.
        Computes counts

        Parameters
        ----------
        Pltrue_M_z: numpy.ndarray
            Probability of lambda true given M_true and z_true

        Returns
        -------
        nc_zbin_lbin: numpy.ndarray
            Number counts in richness and redshift bins
        aux: dict
            Dictionary with intermidate products that can be used for other computations.
            Contains:

                * Plob_M_z (numpy.ndarray): Probability of observed richness in redshift and true mass bins
                * dV_dzob (numpy.ndarray): Array of bin volumes in redshift and richness
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        # intermediate quantity, not returned
        _n_lbdobs_z = np.zeros(lambda_obs_bins_size, dtype=list)

        # outputs
        Plob_M_z = np.zeros(lambda_obs_bins_size, dtype=list)
        dV_dzob = np.zeros((z_obs_bins_size, lambda_obs_bins_size), dtype=list)
        nc_zbin_lbin = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        for ind_lambda in range(lambda_obs_bins_size):

            Plob_M_z[ind_lambda] = self._compute_Plob_M_z_bin(
                lambda_obs_bins[ind_lambda],
                lambda_obs_bins[ind_lambda + 1],
                self.l_m_tab_sig[ind_lambda],
            )
            # N(lob,ztr)
            _n_lbdobs_z[ind_lambda] = simps(
                Plob_M_z[ind_lambda] * self.dndm_z, x=self.integ_mass_arr, axis=1
            )

            for ind_z in range(z_obs_bins_size):

                dV_dzob[ind_z, ind_lambda] = self._compute_volume_bin(
                    z_obs_bins[ind_z],
                    z_obs_bins[ind_z + 1],
                    lambda_obs_bins[ind_lambda],
                )
                nc_zbin_lbin[ind_z, ind_lambda] = self._compute_counts_bin(
                    dV_dzob[ind_z, ind_lambda], _n_lbdobs_z[ind_lambda]
                )

        aux = {"Plob_M_z": Plob_M_z, "dV_dzob": dV_dzob}
        return nc_zbin_lbin, aux

    # -------------------
    # cluster counts cov
    # -------------------

    def _compute_bias(self, Plob_M_z, dV_dzob):
        """compute bias.
        Compute halo bias used in counts covariance in bins of redshift and richness

        Parameters
        ----------
        Plob_M_z : numpy.ndarray
            probability of observed richness given true mass and redshift
        dV_dzob : numpy.ndarray
            array of bin volumes in redshift and richness

        Returns
        -------
        numpy.ndarray
            halo bias in bins of z and lambda
        """

        z_obs_bins_size, lambda_obs_bins_size = dV_dzob.shape

        hbias_zbin_lbin = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        for ind_lambda in range(lambda_obs_bins_size):

            # N(lob,ztr) * bias(lob,ztr)
            _b_n_lbdobs_z = simps(
                Plob_M_z[ind_lambda] * self.dndm_z * self.bias_z,
                x=self.integ_mass_arr,
                axis=1,
            )

            for ind_z in range(z_obs_bins_size):

                # N(lob,zob) * bias(lob,zob)
                hbias_zbin_lbin[ind_z, ind_lambda] = simps(
                    _b_n_lbdobs_z * dV_dzob[ind_z, ind_lambda],
                    x=self.integ_ztrue_arr,
                    axis=0,
                )

        return hbias_zbin_lbin

    def _compute_sab(self, z_obs_bins):

        z_obs_bins_size = len(z_obs_bins) - 1
        z_mid = 0.5 * (z_obs_bins[1:] + z_obs_bins[:-1])

        # initialization
        sab = np.zeros((z_obs_bins_size, z_obs_bins_size))
        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()
        # self.rint = np.zeros((z_obs_bins_size,len(self.integ_k_arr),L+1))

        # power spectrum at the center of observed redshift bins
        pk = self.halo_statistics.matter_power_spectrum(z_mid, self.integ_k_arr)

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        photoz_corr0 = self.photoz_rsd_correction(z_mid, 0)[
            0
        ]  # can neglect richness dependence here
        pk *= photoz_corr0

        # richness-independent quantites that can be computed outside the lambda loop
        for ind_z in range(z_obs_bins_size):

            z_tab = np.linspace(
                z_obs_bins[ind_z], z_obs_bins[ind_z + 1], self.z_tab_sig
            )

            sab[ind_z, : (ind_z + 1)] = (
                1
                / (2 * np.pi**2)
                * simps(
                    (self.integ_k_arr**2 * np.sqrt(pk[ind_z] * pk[: (ind_z + 1)]))
                    * self.covariance.cov_window(ind_z, z_tab, KL),
                    x=self.integ_k_arr,
                    axis=-1,
                )
            )
            sab[: (ind_z + 1), ind_z] = sab[ind_z, : (ind_z + 1)]

        return sab

    def compute_cov(self, z_obs_bins, nc_zbin_lbin, Plob_M_z, dV_dzob):
        """compute counts cov.
        computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        hbias_zbin_lbin : numpy.ndarray
            halo bias, computed with compute_bias function

        Returns
        -------
        numpy.ndarray
            covariance array
        """

        hbias_zbin_lbin = self._compute_bias(Plob_M_z, dV_dzob)
        sab = self._compute_sab(z_obs_bins)

        # shot noise
        _shot_noise = (
            np.diag(nc_zbin_lbin.flatten())
            .reshape(*nc_zbin_lbin.shape, *nc_zbin_lbin.shape)
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance
        cov_nc_zbin_lbin = _shot_noise + (
            hbias_zbin_lbin[np.newaxis, :, np.newaxis, :]
            * hbias_zbin_lbin[:, np.newaxis, :, np.newaxis]
            * sab[:, :, np.newaxis, np.newaxis]
        )

        return cov_nc_zbin_lbin

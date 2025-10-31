# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.hmf_bias import HMFBias
from cloelib.observables.clusters.profile import Profile
from cloelib.observables.clusters.selection_function import SelectionFunction

# import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""


class ClusterCountsStatistics:
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


class ClusterWLStatistics:
    def __init__(
        self,
        cluster_counts_statistics: ClusterCountsStatistics,
        profile: Profile,
        halo_concentration: float,
    ):
        # cluster counts statistics, for integration tables
        self.cluster_counts_statistics = cluster_counts_statistics

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

    def compute_binned_properties(
        self, z_obs_bins, lambda_obs_bins, radius_bins, nc_zbin_lbin, Plob_M_z, dV_dzob
    ):
        """compute reduced shear.
        returns reduced shear array

        Parameters
        ----------
        nc_zbin_lbin : numpy.ndarray
            cluster number counts
        Plob_M_z: numpy.ndarray
            probablity of observed richness given mass and redshift
        dV_dzob: numpy.ndarray
            volume element

        Returns
        -------
        numpy.ndarray
            reduced shear array
        """
        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        z_obs_bins_size = len(z_obs_bins) - 1
        radius_bins_size = len(radius_bins) - 1

        gt_zbin_lbin_rbin = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, radius_bins_size)
        )

        for ind_radius in range(radius_bins_size):
            excess_surface_mass_density = self.profile.excess_surface_mass_density(
                np.atleast_1d(radius_bins[ind_radius]),
                self.cluster_counts_statistics.integ_ztrue_arr,
                self.cluster_counts_statistics.integ_mass_arr,
                self.halo_concentration,
            )
            for ind_lambda in range(lambda_obs_bins_size):
                excesssurfacemassdensity = simps(
                    Plob_M_z[ind_lambda]
                    * self.cluster_counts_statistics.dndm_z
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.cluster_counts_statistics.integ_mass_arr,
                    axis=1,
                )

                for ind_z in range(z_obs_bins_size):
                    gt_zbin_lbin_rbin[ind_z, ind_lambda, ind_radius] = (
                        (1.0)
                        / nc_zbin_lbin[ind_z, ind_lambda]
                        * simps(
                            self.profile.m_sig_crit_m1(
                                self.cluster_counts_statistics.integ_ztrue_arr, ind_z
                            )
                            * dV_dzob[ind_z, ind_lambda]
                            * excesssurfacemassdensity,
                            x=self.cluster_counts_statistics.integ_ztrue_arr,
                        )
                    )
        return gt_zbin_lbin_rbin


class ClusterXi2Statistics:
    def __init__(
        self,
        cluster_counts_statistics: ClusterCountsStatistics,
        clustering: HaloClustering,
    ):
        # cluster counts statistics, for integration tables
        self.cluster_counts_statistics = cluster_counts_statistics

        # observable objects
        self.clustering = clustering

        # hardcoded quantities
        self.l_m_tab_sig_xi2 = [31, 51]
        self.z_tab_sig = 31

    def _compute_pk_ir_resummation(self, z_obs_bins, lambda_obs_bins):

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        _volume_zob = np.zeros(z_obs_bins_size)
        _one_over_n_lambdai_lambdaj = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
            )
        )

        lambda_obs_xi2_mid = 0.5 * (lambda_obs_bins[1:] + lambda_obs_bins[:-1])

        sqrt_Pk_zbin_lbin = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                len(self.cluster_counts_statistics.integ_k_arr),
            )
        )

        ############
        #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
        ############
        pk_IR = self.cluster_counts_statistics.halo_statistics.matter_power_spectrum(
            self.cluster_counts_statistics.integ_ztrue_arr,
            self.cluster_counts_statistics.integ_k_arr,
        )

        # LOOP OVER CLUSTERING RICHNESS BINS
        for ind_lambda in range(lambda_obs_bins_size):

            l_tab = np.geomspace(
                lambda_obs_bins[ind_lambda],
                lambda_obs_bins[ind_lambda + 1],
                self.l_m_tab_sig_xi2[ind_lambda],
            )

            ## recompute quantities that depend on lambda_obs

            # P(lob|ltr,ztr)
            Plob_l_z_xi2 = simps(
                self.cluster_counts_statistics.selectionfunction.P_lbdobs_lbd(
                    self.cluster_counts_statistics.integ_ztrue_arr,
                    self.cluster_counts_statistics.integ_lambda_arr,
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )

            # P(lob|M,ztr)
            Plob_M_z_xi2 = simps(
                self.cluster_counts_statistics.Pltrue_M_z[:, :, :]
                * Plob_l_z_xi2[:, np.newaxis, :],
                x=self.cluster_counts_statistics.integ_lambda_arr,
                axis=-1,
            )

            # n(lob,ztr)
            n_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2 * self.cluster_counts_statistics.dndm_z,
                x=self.cluster_counts_statistics.integ_mass_arr,
                axis=1,
            )

            # n(lob,ztr) * b(lob,zob)
            n_b_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2
                * self.cluster_counts_statistics.dndm_z
                * self.cluster_counts_statistics.bias_z,
                x=self.cluster_counts_statistics.integ_mass_arr,
                axis=1,
            )

            # effective halo bias
            b_eff = (n_b_lbdobs_z_xi2 / n_lbdobs_z_xi2)[:, np.newaxis]

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.cluster_counts_statistics.photoz_rsd_correction(
                    self.cluster_counts_statistics.integ_ztrue_arr,
                    lambda_obs_xi2_mid[ind_lambda],
                )
            )
            pk_halo = pk_IR * (
                b_eff**2 * photoz_corr0 + b_eff * photoz_corr1 + photoz_corr2
            )

            for ind_z in range(z_obs_bins_size):

                z_tab = np.linspace(
                    z_obs_bins[ind_z], z_obs_bins[ind_z + 1], self.z_tab_sig
                )

                # P(zob|ztr)
                Pzob_z = simps(
                    self.cluster_counts_statistics.selectionfunction.P_zobs_z(
                        z_tab,
                        lambda_obs_bins[ind_lambda],
                        self.cluster_counts_statistics.integ_ztrue_arr,
                    ),
                    x=z_tab,
                    axis=0,
                )

                # observed volume element dV/dz_ob
                dV_dzob = (
                    self.cluster_counts_statistics.dvdzdomega_z1z2
                    * Pzob_z
                    * (self.cluster_counts_statistics.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )

                # volume of the observed redshift slice
                _volume_zob[ind_z] = simps(
                    dV_dzob, x=self.cluster_counts_statistics.integ_ztrue_arr, axis=0
                )

                # normalization factor
                N_int_lbdobs_z_xi2 = simps(
                    dV_dzob * n_lbdobs_z_xi2,
                    x=self.cluster_counts_statistics.integ_ztrue_arr,
                    axis=0,
                )

                # power spectrum and shot-noise terms
                sqrt_Pk_zbin_lbin[ind_z, ind_lambda, :] = (
                    simps(
                        (dV_dzob * n_lbdobs_z_xi2)[:, np.newaxis] * np.sqrt(pk_halo),
                        x=self.cluster_counts_statistics.integ_ztrue_arr,
                        axis=0,
                    )
                    / N_int_lbdobs_z_xi2
                )

                _one_over_n_lambdai_lambdaj[ind_z, ind_lambda, ind_lambda] = (
                    _volume_zob[ind_z] / N_int_lbdobs_z_xi2
                )

        # cross Pk and shot-noise in two richness bins
        _Pk_lambdai_lambdaj = (
            sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )  # dim = [nz,nl,nl,nk]
        _one_over_n_lambdai_lambdaj = _one_over_n_lambdai_lambdaj[
            :, :, :, np.newaxis
        ]  # dim = [nz,nl,nl,nk]

        return _Pk_lambdai_lambdaj, _volume_zob, _one_over_n_lambdai_lambdaj

    def _compute_xi2(self, window_radial, Pk_lambdai_lambdaj):

        lambda_obs_bins_size = Pk_lambdai_lambdaj.shape[1]

        # compute 2point correlation function
        # dim = [nz,nl,nl,nr]
        xi2_zbin_lbin_rbin_buf = simps(
            (
                self.cluster_counts_statistics.integ_k_arr**2.0
                / (2.0 * np.pi**2)
                * window_radial[:, np.newaxis, np.newaxis, :, :]
                * Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            ),
            x=self.cluster_counts_statistics.integ_k_arr,
            axis=-1,
        )

        # xi(lambda_i, lambda_j) = xi(lambda_j, lambda_i),  so reshape and keep only one of them

        triangle_indexes = np.triu_indices(lambda_obs_bins_size)
        return xi2_zbin_lbin_rbin_buf[:, triangle_indexes[0], triangle_indexes[1], :]

    def compute_binned_properties(self, z_obs_bins, lambda_obs_bins, radius_bins):
        """Computes xi2

        Parameters
        ----------

        Returns
        -------
        xi2_zbin_lbin_rbin: numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        aux: dict
            Dictionary with intermidate products that can be used for other computations.
            Contains:

                * Pk_lambdai_lambdaj (numpy.ndarray): blah
                * one_over_n_lambdai_lambdaj (numpy.ndarray): blah
                * window_radial (numpy.ndarray): blah
                * volume_radial (numpy.ndarray): blah
                * volume_zob (numpy.ndarray): blah
        """

        # this is never used
        ###integ_zbin_lbin  = np.zeros(((z_obs_bins_size, lambda_obs_bins_size, len(self.cluster_counts_statistics.integ_k_arr))))

        # matter power spectrum + IR resummation
        Pk_lambdai_lambdaj, volume_zob, one_over_n_lambdai_lambdaj = (
            self._compute_pk_ir_resummation(z_obs_bins, lambda_obs_bins)
        )

        # spherical shell window function W(R,K) and volume of the shell (compute once outside the redshift loop)
        # these are used by covariance
        window_radial, volume_radial = self.clustering.WF_ra(
            0.5 * (z_obs_bins[1:] + z_obs_bins[:-1]),
            radius_bins,
        )

        xi2_zbin_lbin_rbin = self._compute_xi2(window_radial, Pk_lambdai_lambdaj)

        aux = {
            "Pk_lambdai_lambdaj": Pk_lambdai_lambdaj,
            "one_over_n_lambdai_lambdaj": one_over_n_lambdai_lambdaj,
            "window_radial": window_radial,
            "volume_radial": volume_radial,
            "volume_zob": volume_zob,
        }
        return xi2_zbin_lbin_rbin, aux

    # ----------------------
    # clustering covariance
    # ----------------------

    def _compute_alpha_beta(
        self,
        alpha_cov_xi2,
        beta_cov_xi2,
        Pk_lambdai_lambdaj,
        one_over_n_lambdai_lambdaj,
    ):
        # combine and reshape
        alpha_ij = (1 + alpha_cov_xi2[:, :, np.newaxis, np.newaxis]) * (
            1 + alpha_cov_xi2[:, np.newaxis, :, np.newaxis]
        )
        beta_ij = (
            beta_cov_xi2[:, :, np.newaxis, np.newaxis]
            * beta_cov_xi2[:, np.newaxis, :, np.newaxis]
        )

        beta_pk_ij = beta_ij * Pk_lambdai_lambdaj
        alpha_n_ij = alpha_ij * one_over_n_lambdai_lambdaj

        return alpha_n_ij, beta_pk_ij

    def compute_cov(
        self,
        Pk_lambdai_lambdaj,
        one_over_n_lambdai_lambdaj,
        window_radial,
        volume_radial,
        volume_zob,
    ):
        z_obs_bins_size, lambda_obs_bins_size, _, _ = one_over_n_lambdai_lambdaj.shape
        _, radius_bins_size = volume_radial.shape
        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0,beta=1,gamma=0 (see Euclid Collaboration:
        #    Fumagalli et al. 2022)
        #    cov_g, cov_ng are TWO TERMS OF EQ. 73
        _alpha_cov_xi2 = np.zeros((z_obs_bins_size, lambda_obs_bins_size))
        _beta_cov_xi2 = np.ones((z_obs_bins_size, lambda_obs_bins_size))
        gamma_cov_xi2 = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        # compute nuisance parameters
        alpha_n_ij, beta_pk_ij = self._compute_alpha_beta(
            _alpha_cov_xi2,
            _beta_cov_xi2,
            Pk_lambdai_lambdaj,
            one_over_n_lambdai_lambdaj,
        )

        # internal attributes
        _cov_g = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                radius_bins_size,
                radius_bins_size,
            )
        )
        _cov_ng = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
                radius_bins_size,
                radius_bins_size,
            )
        )

        # define cluster clustering bin numbers for loops
        z_bin_numbers = range(z_obs_bins_size)
        lambda_bin_numbers = range(lambda_obs_bins_size)
        rad_bin_numbers = range(radius_bins_size)

        # note: this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_numbers:
            for ind_lambda_j in lambda_bin_numbers:
                for ind_radius in rad_bin_numbers:

                    _cov_ng[
                        :,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_radius,
                        ind_radius,
                    ] = (
                        simps(
                            self.cluster_counts_statistics.integ_k_arr**2.0
                            / (2.0 * np.pi**2.0)
                            * window_radial[:, ind_radius, :]
                            * beta_pk_ij[:, ind_lambda_i, ind_lambda_j, :],
                            x=self.cluster_counts_statistics.integ_k_arr,
                        )
                        * (1 + gamma_cov_xi2[:, ind_lambda_i])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_i, ind_lambda_i, 0]
                        * (1 + gamma_cov_xi2[:, ind_lambda_j])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_j, ind_lambda_j, 0]
                        / volume_radial[:, ind_radius]
                    )

                    for ind_lambda_k in lambda_bin_numbers:
                        for ind_lambda_h in lambda_bin_numbers:

                            # gaussian term
                            _cov_g[
                                :,
                                ind_lambda_i,
                                ind_lambda_j,
                                ind_lambda_k,
                                ind_lambda_h,
                                :,
                                :,
                            ] = simps(
                                self.cluster_counts_statistics.integ_k_arr**2.0
                                / (2.0 * np.pi**2.0)
                                * window_radial[:, np.newaxis, :, :]
                                * window_radial[:, :, np.newaxis, :]
                                * (beta_pk_ij + alpha_n_ij)[
                                    :,
                                    ind_lambda_i,
                                    ind_lambda_k,
                                    np.newaxis,
                                    np.newaxis,
                                    :,
                                ]
                                * (beta_pk_ij + alpha_n_ij)[
                                    :,
                                    ind_lambda_j,
                                    ind_lambda_h,
                                    np.newaxis,
                                    np.newaxis,
                                    :,
                                ],
                                x=self.cluster_counts_statistics.integ_k_arr,
                                axis=-1,
                            )

        # Compute the covariance
        cov_xi2_zbin_lbin_rbin = (
            (_cov_g + _cov_ng)
            + (_cov_g + _cov_ng).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_xi2 bins
            )
        ) / volume_zob[
            :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_i, lambda_j, lambda_k, lambda_l) = cov_xi(lambda_j, lambda_i, lambda_l, lambda_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_bins_size)
        # simplify first pair
        cov_xi2_zbin_lbin_rbin = cov_xi2_zbin_lbin_rbin[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        cov_xi2_zbin_lbin_rbin = cov_xi2_zbin_lbin_rbin[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (nz, nl_red, nl_red, nrad, nrad),
        # make it (nz, nz, nl_red, nl_red, nrad, nrad), being diagonal in (nz, nz)
        # """
        cov_xi2_zbin_lbin_rbin = (
            np.diag(np.ones(z_obs_bins_size))[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * cov_xi2_zbin_lbin_rbin
        )
        return cov_xi2_zbin_lbin_rbin

# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.summary_statistics.clusters.counts import ClusterCounts

# import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""


class ClusterXi2:
    def __init__(
        self,
        integ_tables: dict,
        halo_statistics: HaloStatistics,
        clustering: HaloClustering,
        area: float = 10313,
    ):
        # integration tables
        self.integ_tables = integ_tables

        # observable objects
        self.halo_statistics = halo_statistics
        self.clustering = clustering

        # internal values
        self.area = area

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
                len(self.integ_tables["k"]),
            )
        )

        ############
        #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
        ############
        pk_IR = self.halo_statistics.matter_power_spectrum(
            self.integ_tables["ztrue"],
            self.integ_tables["k"],
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
                self.clustering.selectionfunction.P_lbdobs_lbd(
                    self.integ_tables["ztrue"],
                    self.integ_tables["lambda"],
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )

            # P(lob|M,ztr)
            Plob_M_z_xi2 = simps(
                self.integ_tables["Pltrue_M_z"][:, :, :]
                * Plob_l_z_xi2[:, np.newaxis, :],
                x=self.integ_tables["lambda"],
                axis=-1,
            )

            # n(lob,ztr)
            n_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2 * self.integ_tables["dndm_z"],
                x=self.integ_tables["mass"],
                axis=1,
            )

            # n(lob,ztr) * b(lob,zob)
            n_b_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2
                * self.integ_tables["dndm_z"]
                * self.integ_tables["bias_z"],
                x=self.integ_tables["mass"],
                axis=1,
            )

            # effective halo bias
            b_eff = (n_b_lbdobs_z_xi2 / n_lbdobs_z_xi2)[:, np.newaxis]

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.clustering.photoz_rsd_correction(
                    self.integ_tables["ztrue"],
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
                    self.clustering.selectionfunction.P_zobs_z(
                        z_tab,
                        lambda_obs_bins[ind_lambda],
                        self.integ_tables["ztrue"],
                    ),
                    x=z_tab,
                    axis=0,
                )

                # observed volume element dV/dz_ob
                dV_dzob = (
                    self.integ_tables["dvdzdomega_z1z2"]
                    * Pzob_z
                    * (self.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )

                # volume of the observed redshift slice
                _volume_zob[ind_z] = simps(
                    dV_dzob, x=self.integ_tables["ztrue"], axis=0
                )

                # normalization factor
                N_int_lbdobs_z_xi2 = simps(
                    dV_dzob * n_lbdobs_z_xi2,
                    x=self.integ_tables["ztrue"],
                    axis=0,
                )

                # power spectrum and shot-noise terms
                sqrt_Pk_zbin_lbin[ind_z, ind_lambda, :] = (
                    simps(
                        (dV_dzob * n_lbdobs_z_xi2)[:, np.newaxis] * np.sqrt(pk_halo),
                        x=self.integ_tables["ztrue"],
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
                self.integ_tables["k"] ** 2.0
                / (2.0 * np.pi**2)
                * window_radial[:, np.newaxis, np.newaxis, :, :]
                * Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            ),
            x=self.integ_tables["k"],
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
        ###integ_zbin_lbin  = np.zeros(((z_obs_bins_size, lambda_obs_bins_size, len(self.integ_tables["k"]))))

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
        z_bin_loop = range(z_obs_bins_size)
        lambda_bin_loop = range(lambda_obs_bins_size)
        rad_bin_loop = range(radius_bins_size)

        # note: this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_loop:
            for ind_lambda_j in lambda_bin_loop:
                for ind_radius in rad_bin_loop:

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
                            self.integ_tables["k"] ** 2.0
                            / (2.0 * np.pi**2.0)
                            * window_radial[:, ind_radius, :]
                            * beta_pk_ij[:, ind_lambda_i, ind_lambda_j, :],
                            x=self.integ_tables["k"],
                        )
                        * (1 + gamma_cov_xi2[:, ind_lambda_i])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_i, ind_lambda_i, 0]
                        * (1 + gamma_cov_xi2[:, ind_lambda_j])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_j, ind_lambda_j, 0]
                        / volume_radial[:, ind_radius]
                    )

                for ind_lambda_k in lambda_bin_loop:
                    for ind_lambda_h in lambda_bin_loop:

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
                            self.integ_tables["k"] ** 2.0
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
                            x=self.integ_tables["k"],
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

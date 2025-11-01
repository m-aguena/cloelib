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
        self.l_m_tab_sig = [31, 51]
        self.z_tab_sig = 31

    def _compute_pk_ir_resummation(self, z_obs_bins, lambda_obs_bins):
        """Computes Pk IR resummation.

        Parameters
        ----------
        z_obs_bins: numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins: numpy.ndarray
            Richness bins for the integration.

        Returns
        -------
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???
        volume_zob : numpy.ndarray
            Observed volume in each redshift bin
        one_over_n_lambdai_lambdaj : numpy.ndarray
            volume_zob / nc_int_lbdobs_z in each redshift bin
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        volume_zob = np.zeros(z_obs_bins_size)
        one_over_n_lambdai_lambdaj = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                lambda_obs_bins_size,
            )
        )

        lambda_obs_mid = 0.5 * (lambda_obs_bins[1:] + lambda_obs_bins[:-1])

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
                self.l_m_tab_sig[ind_lambda],
            )

            ## recompute quantities that depend on lambda_obs

            # P(lob|ltr,ztr)
            Plob_l_z = simps(
                self.clustering.selectionfunction.P_lbdobs_lbd(
                    self.integ_tables["ztrue"],
                    self.integ_tables["lambda_true"],
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )

            # P(lob|M,ztr)
            Plob_M_z = simps(
                self.integ_tables["Pltrue_M_z"][:, :, :] * Plob_l_z[:, np.newaxis, :],
                x=self.integ_tables["lambda_true"],
                axis=-1,
            )

            # n(lob,ztr)
            nc_lbdobs_z = simps(
                Plob_M_z * self.integ_tables["dndm_z"],
                x=self.integ_tables["mass"],
                axis=1,
            )

            # n(lob,ztr) * b(lob,zob)
            hbias_lbdobs_z = simps(
                Plob_M_z * self.integ_tables["dndm_z"] * self.integ_tables["bias_z"],
                x=self.integ_tables["mass"],
                axis=1,
            )

            # effective halo bias
            b_eff = (hbias_lbdobs_z / nc_lbdobs_z)[:, np.newaxis]

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.clustering.photoz_rsd_correction(
                    self.integ_tables["ztrue"],
                    lambda_obs_mid[ind_lambda],
                )
            )
            pk_halo = pk_IR * (
                b_eff**2 * photoz_corr0 + b_eff * photoz_corr1 + photoz_corr2
            )

            for ind_z in range(z_obs_bins_size):

                _z_tab = np.linspace(
                    z_obs_bins[ind_z], z_obs_bins[ind_z + 1], self.z_tab_sig
                )

                # P(zob|ztr)
                _Pzob_z = simps(
                    self.clustering.selectionfunction.P_zobs_z(
                        _z_tab,
                        lambda_obs_bins[ind_lambda],
                        self.integ_tables["ztrue"],
                    ),
                    x=_z_tab,
                    axis=0,
                )

                # observed volume element dV/dz_ob
                _dV_dzob = (
                    self.integ_tables["dvdzdomega_z1z2"]
                    * _Pzob_z
                    * (self.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )

                # volume of the observed redshift slice
                volume_zob[ind_z] = simps(
                    _dV_dzob, x=self.integ_tables["ztrue"], axis=0
                )

                # normalization factor
                _nc_int_lbdobs_z = simps(
                    _dV_dzob * nc_lbdobs_z,
                    x=self.integ_tables["ztrue"],
                    axis=0,
                )

                # power spectrum and shot-noise terms
                sqrt_Pk_zbin_lbin[ind_z, ind_lambda, :] = (
                    simps(
                        (_dV_dzob * nc_lbdobs_z)[:, np.newaxis] * np.sqrt(pk_halo),
                        x=self.integ_tables["ztrue"],
                        axis=0,
                    )
                    / _nc_int_lbdobs_z
                )

                one_over_n_lambdai_lambdaj[ind_z, ind_lambda, ind_lambda] = (
                    volume_zob[ind_z] / _nc_int_lbdobs_z
                )

        # cross Pk and shot-noise in two richness bins
        Pk_lambdai_lambdaj = (
            sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )  # dim = [nz,nl,nl,nk]
        one_over_n_lambdai_lambdaj = one_over_n_lambdai_lambdaj[
            :, :, :, np.newaxis
        ]  # dim = [nz,nl,nl,nk]

        return Pk_lambdai_lambdaj, volume_zob, one_over_n_lambdai_lambdaj

    def _compute_xi2(self, window_radial, Pk_lambdai_lambdaj):
        """Computes xi2

        Parameters
        ----------
        window_radial : numpy.ndarray
            blah
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???

        Returns
        -------
        xi2_zbin_lbin_rbin: numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        """

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

    def compute_binned_quantities(self, z_obs_bins, lambda_obs_bins, radius_bins):
        """Computes binned quantities (xi2+aux)

        Parameters
        ----------

        Returns
        -------
        xi2_zbin_lbin_rbin: numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        aux: dict
            Dictionary with intermidate products that can be used for other computations.
            Contains:

                * Pk_lambdai_lambdaj (numpy.ndarray): Power spectrum ???
                * one_over_n_lambdai_lambdaj (numpy.ndarray): volume_zob / nc_int_lbdobs_z in each redshift bin
                * window_radial (numpy.ndarray): blah
                * volume_radial (numpy.ndarray): blah
                * volume_zob (numpy.ndarray): Observed volume in each redshift bin
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
        alpha,
        beta,
        Pk_lambdai_lambdaj,
        one_over_n_lambdai_lambdaj,
    ):
        """Computes mean alpha and mean beta*Pk  in richness bins

        Parameters
        ----------
        alpha: numpy.ndarray
            Alpha parameter
        beta: numpy.ndarray
            Beta parameter

        Returns
        -------
        alpha_n_ij: numpy.ndarray
            Mean alpha in richness bins
        beta_pk_ij: numpy.ndarray
            Mean beta*Pk in richness bins
        """
        # combine and reshape
        alpha_ij = (1 + alpha[:, :, np.newaxis, np.newaxis]) * (
            1 + alpha[:, np.newaxis, :, np.newaxis]
        )
        beta_ij = (
            beta[:, :, np.newaxis, np.newaxis] * beta[:, np.newaxis, :, np.newaxis]
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
        """Computes xi2 covariance.

        Parameters
        ----------
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???
        one_over_n_lambdai_lambdaj : numpy.ndarray
            volume_zob / nc_int_lbdobs_z in each redshift bin
        window_radial : numpy.ndarray
            blah
        volume_radial : numpy.ndarray
            blah
        volume_zob : numpy.ndarray
            Observed volume in each redshift bin

        Returns
        -------
        cov_xi2_zbin_lbin_rbin: numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_bins_size, lambda_obs_bins_size, _, _ = one_over_n_lambdai_lambdaj.shape
        _, radius_bins_size = volume_radial.shape

        # compute nuisance parameters

        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0,beta=1,gamma=0 (see Euclid Collaboration:
        #    Fumagalli et al. 2022)
        alpha_n_ij, beta_pk_ij = self._compute_alpha_beta(
            alpha=np.zeros((z_obs_bins_size, lambda_obs_bins_size)),
            beta=np.ones((z_obs_bins_size, lambda_obs_bins_size)),
            Pk_lambdai_lambdaj=Pk_lambdai_lambdaj,
            one_over_n_lambdai_lambdaj=one_over_n_lambdai_lambdaj,
        )
        gamma = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        # cov_g, cov_ng are TWO TERMS OF EQ. 73
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
                        * (1 + gamma[:, ind_lambda_i])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_i, ind_lambda_i, 0]
                        * (1 + gamma[:, ind_lambda_j])
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

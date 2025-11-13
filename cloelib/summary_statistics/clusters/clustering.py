# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.summary_statistics.clusters.counts import ClusterCounts

# import jax

"""

## Notes :

- Clusters clustering class

"""


class ClusterClustering:
    def __init__(
        self,
        cluster_counts: ClusterCounts,
        clustering: HaloClustering,
        area: float = 10313,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_counts : ClusterCounts
            Cluster counts summary statistics object
        clustering : HaloClustering
            Halo clustering object
        area : float
            Area of the survey in deg2.
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_counts = cluster_counts

        # observable objects
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
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins : numpy.ndarray
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

        # output
        volume_zob = np.zeros(z_obs_bins_size)

        # intermediate quantites
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
                len(self.cluster_counts.cluster_stat_kernel_tables["k"]),
            )
        )

        # get cluster counts quantities
        nc_zbin_lbin, counts_intermediate_products_zbin_lbin = (
            self.cluster_counts.compute_binned_counts(
                z_obs_bins=z_obs_bins,
                lambda_obs_bins=lambda_obs_bins,
                z_tab_sig=self.z_tab_sig,
                l_m_tab_sig=self.l_m_tab_sig,
            )
        )
        _, bias_intermediate_products_zbin_lbin = (
            self.cluster_counts.compute_binned_bias(
                counts_intermediate_products_zbin_lbin["Plob_M_z"],
                counts_intermediate_products_zbin_lbin["dV_dzob"],
            )
        )
        # effective halo bias
        b_eff = (
            bias_intermediate_products_zbin_lbin["hb_lbdobs_z"]
            / counts_intermediate_products_zbin_lbin["nc_lbdobs_z"]
        )

        ############
        #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
        ############
        pk_IR = self.cluster_counts.halo_statistics.matter_power_spectrum(
            self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
            self.cluster_counts.cluster_stat_kernel_tables["k"],
        )

        # Compute intermediate quantities
        for ind_lambda in range(lambda_obs_bins_size):

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)

            # effective halo bias
            _b_eff = b_eff[ind_lambda][:, np.newaxis]
            # rsd corrections
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.clustering.photoz_rsd_correction(
                    self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                    lambda_obs_mid[ind_lambda],
                )
            )
            # corrected power specrum
            pk_halo = pk_IR * (
                _b_eff**2 * photoz_corr0 + _b_eff * photoz_corr1 + photoz_corr2
            )

            for ind_z in range(z_obs_bins_size):

                # volume of the observed redshift slice
                volume_zob[ind_z] = simps(
                    counts_intermediate_products_zbin_lbin["dV_dzob"][
                        ind_z, ind_lambda
                    ],
                    x=self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                    axis=0,
                )

                # power spectrum and shot-noise terms
                sqrt_Pk_zbin_lbin[ind_z, ind_lambda, :] = (
                    simps(
                        (
                            counts_intermediate_products_zbin_lbin["dV_dzob"][
                                ind_z, ind_lambda
                            ]
                            * counts_intermediate_products_zbin_lbin["nc_lbdobs_z"][
                                ind_lambda
                            ]
                        )[:, np.newaxis]
                        * np.sqrt(pk_halo),
                        x=self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                        axis=0,
                    )
                    / nc_zbin_lbin[ind_z, ind_lambda]
                )

                one_over_n_lambdai_lambdaj[ind_z, ind_lambda, ind_lambda] = (
                    volume_zob[ind_z] / nc_zbin_lbin[ind_z, ind_lambda]
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

    def _compute_clustering(self, shell_window, Pk_lambdai_lambdaj):
        """Computes the 3D two-point correlation function.

        Parameters
        ----------
        shell_window : numpy.ndarray
            Cluster count covariance window (i,j,k) where i is the redshift bin, j is the radial bin and k are the wavenumbers
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???

        Returns
        -------
        clustering_zbin_lbin_rbin : numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        """

        lambda_obs_bins_size = Pk_lambdai_lambdaj.shape[1]

        # compute 2point correlation function
        # dim = [nz,nl,nl,nr]
        clustering_zbin_lbin_rbin_buf = simps(
            (
                self.cluster_counts.cluster_stat_kernel_tables["k"] ** 2.0
                / (2.0 * np.pi**2)
                * shell_window[:, np.newaxis, np.newaxis, :, :]
                * Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            ),
            x=self.cluster_counts.cluster_stat_kernel_tables["k"],
            axis=-1,
        )

        # xi(lambda_i, lambda_j) = xi(lambda_j, lambda_i),  so reshape and keep only one of them

        triangle_indexes = np.triu_indices(lambda_obs_bins_size)
        return clustering_zbin_lbin_rbin_buf[
            :, triangle_indexes[0], triangle_indexes[1], :
        ]

    def compute_binned_clustering(
        self,
        z_obs_bins,
        lambda_obs_bins,
        radius_bins,
        return_intermediate_products=True,
    ):
        """Computes binned quantities (clustering+aux)

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.
        radius_obs_bins : numpy.ndarray
            Radial bins for the profile.

        Returns
        -------
        clustering_zbin_lbin_rbin : numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        intermediate_products_zbin_lbin (optional) : dict
            Dictionary with intermidate products that can be used for other computations.
            Returned only when `return_intermediate_products` is true.
            Contains :

                * Pk_lambdai_lambdaj (numpy.ndarray) : Power spectrum ???
                * one_over_n_lambdai_lambdaj (numpy.ndarray) : volume_zob / nc_int_lbdobs_z in each redshift bin
                * shell_window (numpy.ndarray) : Cluster count covariance window (i,j,k) where i is the redshift bin, j is the radial bin and k are the wavenumbers
                * shell_volume (numpy.ndarray) : Spherical shell volume (i,j) where i is the redshift bin and j is the radial bin
                * volume_zob (numpy.ndarray) : Observed volume in each redshift bin
        """
        # matter power spectrum + IR resummation
        Pk_lambdai_lambdaj, volume_zob, one_over_n_lambdai_lambdaj = (
            self._compute_pk_ir_resummation(z_obs_bins, lambda_obs_bins)
        )

        # spherical shell window function W(R,K) and volume of the shell (compute once outside the redshift loop)
        # these are used by covariance
        shell_window, shell_volume = self.clustering.WF_ra(
            0.5 * (z_obs_bins[1:] + z_obs_bins[:-1]),
            radius_bins,
        )

        clustering_zbin_lbin_rbin = self._compute_clustering(
            shell_window, Pk_lambdai_lambdaj
        )

        intermediate_products_zbin_lbin = {
            "Pk_lambdai_lambdaj": Pk_lambdai_lambdaj,
            "one_over_n_lambdai_lambdaj": one_over_n_lambdai_lambdaj,
            "shell_window": shell_window,
            "shell_volume": shell_volume,
            "volume_zob": volume_zob,
        }
        return clustering_zbin_lbin_rbin, intermediate_products_zbin_lbin

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
        alpha : numpy.ndarray
            Alpha parameter
        beta : numpy.ndarray
            Beta parameter

        Returns
        -------
        alpha_n_ij : numpy.ndarray
            Mean alpha in richness bins
        beta_pk_ij : numpy.ndarray
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
        shell_window,
        shell_volume,
        volume_zob,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        one_over_n_lambdai_lambdaj : numpy.ndarray
            volume_zob / nc_int_lbdobs_z in each redshift bin.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        shell_window : numpy.ndarray
            Cluster count covariance window (i,j,k) where i is the redshift bin,
            j is the radial bin and k are the wavenumbers.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        shell_volume : numpy.ndarray
            Spherical shell volume (i,j) where i is the redshift bin and j is the radial bin.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        volume_zob : numpy.ndarray
            Observed volume in each redshift bin.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.

        Returns
        -------
        cov_clustering_zbin_lbin_rbin : numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_bins_size, lambda_obs_bins_size, _, _ = one_over_n_lambdai_lambdaj.shape
        _, radius_bins_size = shell_volume.shape

        # compute nuisance parameters

        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0,beta=1,gamma=0 (see Euclid Collaboration :
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

        # note : this could be reduced to compute only half of the matrix
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
                            self.cluster_counts.cluster_stat_kernel_tables["k"] ** 2.0
                            / (2.0 * np.pi**2.0)
                            * shell_window[:, ind_radius, :]
                            * beta_pk_ij[:, ind_lambda_i, ind_lambda_j, :],
                            x=self.cluster_counts.cluster_stat_kernel_tables["k"],
                        )
                        * (1 + gamma[:, ind_lambda_i])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_i, ind_lambda_i, 0]
                        * (1 + gamma[:, ind_lambda_j])
                        * one_over_n_lambdai_lambdaj[:, ind_lambda_j, ind_lambda_j, 0]
                        / shell_volume[:, ind_radius]
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
                            self.cluster_counts.cluster_stat_kernel_tables["k"] ** 2.0
                            / (2.0 * np.pi**2.0)
                            * shell_window[:, np.newaxis, :, :]
                            * shell_window[:, :, np.newaxis, :]
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
                            x=self.cluster_counts.cluster_stat_kernel_tables["k"],
                            axis=-1,
                        )

        # Compute the covariance
        cov_clustering_zbin_lbin_rbin = (
            (_cov_g + _cov_ng)
            + (_cov_g + _cov_ng).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_clustering bins
            )
        ) / volume_zob[
            :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_i, lambda_j, lambda_k, lambda_l) = cov_xi(lambda_j, lambda_i, lambda_l, lambda_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_bins_size)
        # simplify first pair
        cov_clustering_zbin_lbin_rbin = cov_clustering_zbin_lbin_rbin[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        cov_clustering_zbin_lbin_rbin = cov_clustering_zbin_lbin_rbin[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (nz, nl_red, nl_red, nrad, nrad),
        # make it (nz, nz, nl_red, nl_red, nrad, nrad), being diagonal in (nz, nz)
        # """
        cov_clustering_zbin_lbin_rbin = (
            np.diag(np.ones(z_obs_bins_size))[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * cov_clustering_zbin_lbin_rbin
        )
        return cov_clustering_zbin_lbin_rbin

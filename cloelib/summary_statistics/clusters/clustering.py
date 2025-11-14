# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Clusters clustering class

"""


class ClusterClustering:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        clustering: HaloClustering,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        clustering : HaloClustering
            Halo clustering object
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.clustering = clustering

        # hardcoded quantities for integration
        self.l_m_tab_sig = [31, 51]
        self.z_tab_sig = 31

    def _compute_pk_ir_resummation_unnormalized(
        self, lambda_obs_mid, dv_dz_zbin_z, p_lbin_z, b_lbin_z
    ):
        """Computes Pk IR resummation.

        Parameters
        ----------
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.

        Returns
        -------
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???
        """
        # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
        # rsd corrections (l_obs, z, k)
        photoz_corr0, photoz_corr1, photoz_corr2 = np.array(
            [
                self.clustering.photoz_rsd_correction(
                    self.cluster_statitstics_modeling.kernel_tables["ztrue"],
                    _lambda_obs,
                )
                for _lambda_obs in lambda_obs_mid
            ]
        ).transpose(1, 0, 2, 3)

        # compute effective halo bias, with shape (l_obs, z, 1)
        b_eff = (b_lbin_z / p_lbin_z)[:, :, np.newaxis]

        # corrected power specrum (l_obs, z, k)
        pk_halo = (
            b_eff**2 * photoz_corr0 + b_eff * photoz_corr1 + photoz_corr2
        ) * self.cluster_statitstics_modeling.halo_statistics.matter_power_spectrum(
            self.cluster_statitstics_modeling.kernel_tables["ztrue"],
            self.cluster_statitstics_modeling.kernel_tables["k"],
        )

        # average square of power spectrum in redshift and richness bins (z_obs, l_obs, k)
        sqrt_Pk_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            dv_dz_zbin_z[:, :, :, np.newaxis]
            * p_lbin_z[np.newaxis, :, :, np.newaxis]
            * np.sqrt(pk_halo[np.newaxis, :, :, :])
        )

        # Compute output Pk (z_obs, l_obs, l_obs, k)
        Pk_lambdai_lambdaj = (
            sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )
        return Pk_lambdai_lambdaj

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

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # volume element in each redshift and richness bin shape (z_obs, lambda_obs, z)
        dv_dz_zbin_z = self.cluster_statitstics_modeling.compute_binned_volume(
            z_obs_bins, lambda_obs_bins, self.z_tab_sig
        )
        # integral of p_lbin_M_z*dndm_z on mass (l_obs, mass, z)
        _p_lbin_M_z = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        # integral of p_lbin_M_z*dndm_z on mass (l_obs, z)
        p_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                _p_lbin_M_z
            )
        )
        # integral of p_lbin_M_z*dndm_z*bias_z on mass (l_obs, z)
        b_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                _p_lbin_M_z * self.cluster_statitstics_modeling.kernel_tables["bias_z"]
            )
        )
        # cluster counts (z_obs, l_obs)
        nc_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            p_lbin_z[np.newaxis, :] * dv_dz_zbin_z
        )

        ################################################
        # Computes the 3D two-point correlation function
        ################################################

        # matter power spectrum + IR resummation (z_obs, l_obs, l_obs, k)
        _lambda_obs_mid = 0.5 * (lambda_obs_bins[1:] + lambda_obs_bins[:-1])
        Pk_lambdai_lambdaj = self._compute_pk_ir_resummation_unnormalized(
            _lambda_obs_mid, dv_dz_zbin_z, p_lbin_z, b_lbin_z
        ) / (
            nc_zbin_lbin[:, np.newaxis, :, np.newaxis]
            * nc_zbin_lbin[:, :, np.newaxis, np.newaxis]
        )

        # Spherical shell window (z_obs, radius, k) and
        # volume of the shell (z_obs, radius) in each z_obs_bin
        # shell_volume is used only by covariance
        _z_obs_mid = 0.5 * (z_obs_bins[1:] + z_obs_bins[:-1])
        shell_window, shell_volume = self.clustering.WF_ra(_z_obs_mid, radius_bins)

        # compute 2point correlation function (z_obs, l_obs, l_obs, radius)
        _clustering_zbin_lbin_rbin_buf = (
            self.cluster_statitstics_modeling.integrate_quantity_in_k_space(
                shell_window[:, np.newaxis, np.newaxis, :, :]
                * Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            )
        )

        # xi(lambda_i, lambda_j) = xi(lambda_j, lambda_i) so we reshape
        # and keep only one of them, with a (z_obs, l_obs, radius) output
        triangle_indexes = np.triu_indices(len(lambda_obs_bins) - 1)
        clustering_zbin_lbin_rbin = _clustering_zbin_lbin_rbin_buf[
            :, triangle_indexes[0], triangle_indexes[1], :
        ]

        if not return_intermediate_products:
            return clustering_zbin_lbin_rbin

        intermediate_products_zbin_lbin = {
            "Pk_lambdai_lambdaj": Pk_lambdai_lambdaj,
            "shell_window": shell_window,
            "shell_volume": shell_volume,
            "nc_zbin_lbin": nc_zbin_lbin,
            "dv_dz_zbin_z": dv_dz_zbin_z,
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
        shell_window,
        shell_volume,
        dv_dz_zbin_z,
        nc_zbin_lbin,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        Pk_lambdai_lambdaj : numpy.ndarray
            Power spectrum ???
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        shell_window : numpy.ndarray
            Cluster count covariance window (i,j,k) where i is the redshift bin,
            j is the radial bin and k are the wavenumbers.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        shell_volume : numpy.ndarray
            Spherical shell volume (i,j) where i is the redshift bin and j is the radial bin.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        dv_dz_zbin_z : numpy.ndarray
            Observed volume element (dV/dz_ob) in each redshift and richness bin
            shape (z_obs, lambda_obs, z) with (z) in cluster_statitstics_modeling.kenel_tables.
        nc_zbin_lbin : numpy.ndarray
            Number counts in redshift and richness bins

        Returns
        -------
        cov_clustering_zbin_lbin_rbin : numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_bins_size, lambda_obs_bins_size = nc_zbin_lbin.shape
        _, radius_bins_size = shell_volume.shape

        # Compute observed volume in each redshift bin (z_obs, lambda_obs)
        volume_zob = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            dv_dz_zbin_z
        )
        # Compute output shot-noise terms (z_obs, l_obs, l_obs, 1)
        one_over_n_lambdai_lambdaj = (
            volume_zob[:, :, np.newaxis]
            / nc_zbin_lbin[:, :, np.newaxis]
            * np.identity(lambda_obs_bins_size)[np.newaxis, :, :]
        )[:, :, :, np.newaxis]

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
                        self.cluster_statitstics_modeling.integrate_quantity_in_k_space(
                            shell_window[:, ind_radius, :]
                            * beta_pk_ij[:, ind_lambda_i, ind_lambda_j, :],
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
                        ] = self.cluster_statitstics_modeling.integrate_quantity_in_k_space(
                            shell_window[:, np.newaxis, :, :]
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
                            ]
                        )

        # Compute the covariance
        cov_clustering_zbin_lbin_rbin = (
            (_cov_g + _cov_ng)
            + (_cov_g + _cov_ng).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_clustering bins
            )
        ) / volume_zob[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

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
            np.identity(z_obs_bins_size)[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * cov_clustering_zbin_lbin_rbin
        )
        return cov_clustering_zbin_lbin_rbin

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
        self, lambda_obs_mid, dvdz_zbin_lbin_z, p_lbin_z, b_lbin_z
    ):
        """Computes Pk IR resummation.

        Parameters
        ----------
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.

        Returns
        -------
        pk_zbin_lbin_lbin_k : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation),
            NOT normalized by the number counts.
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
            dvdz_zbin_lbin_z[:, :, :, np.newaxis]
            * p_lbin_z[np.newaxis, :, :, np.newaxis]
            * np.sqrt(pk_halo[np.newaxis, :, :, :])
        )

        # Compute output Pk (z_obs, l_obs, l_obs, k)
        pk_zbin_lbin_lbin_k = (
            sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )
        return pk_zbin_lbin_lbin_k

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

                * pk_zbin_lbin_lbin_k (numpy.ndarray) : Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
                * window_zbin_lbin_k (numpy.ndarray) : Cluster count covariance window (z_obs, l_obs, k).
                * vol_zbin_rbin (numpy.ndarray) : Spherical shell volume (z_obs, radius).
                * dvdz_zbin_lbin_z (numpy.ndarray) : Observed volume element (dV/dz_ob) in each redshift and richness bin
                * nc_zbin_lbin (numpy.ndarray) :  Number counts in redshift and richness bins
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # volume element in each redshift and richness bin shape : (z_obs, l_obs, z)
        dvdz_zbin_lbin_z = (
            self.cluster_statitstics_modeling.compute_binned_volume_element(
                z_obs_bins, lambda_obs_bins, self.z_tab_sig
            )
        )
        # P(lambda_obs_bins|M, z) : (l_obs, mass, z)
        _p_lbin_z_m = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        # integral of P(lambda_obs_bins|M, z)*b(z)*dn/dM on mass : (l_obs, z)
        p_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                _p_lbin_z_m
            )
        )
        # integral of P(lambda_obs_bins|M, z)*dn/dM on mass : (l_obs, z)
        b_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                _p_lbin_z_m
                * self.cluster_statitstics_modeling.kernel_tables["bias_z_m"]
            )
        )
        # cluster counts : (z_obs, l_obs)
        nc_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            p_lbin_z[np.newaxis, :] * dvdz_zbin_lbin_z
        )

        ################################################
        # Computes the 3D two-point correlation function
        ################################################

        # matter power spectrum + IR resummation : (z_obs, l_obs, l_obs, k)
        _lambda_obs_mid = 0.5 * (lambda_obs_bins[1:] + lambda_obs_bins[:-1])
        pk_zbin_lbin_lbin_k = self._compute_pk_ir_resummation_unnormalized(
            _lambda_obs_mid, dvdz_zbin_lbin_z, p_lbin_z, b_lbin_z
        ) / (
            nc_zbin_lbin[:, np.newaxis, :, np.newaxis]
            * nc_zbin_lbin[:, :, np.newaxis, np.newaxis]
        )

        # Spherical shell window : (z_obs, radius, k) and
        # volume of the shell : (z_obs, radius) in each z_obs_bin
        # vol_zbin_rbin is used only by covariance
        _z_obs_mid = 0.5 * (z_obs_bins[1:] + z_obs_bins[:-1])
        window_zbin_lbin_k, vol_zbin_rbin = self.clustering.WF_ra(
            _z_obs_mid, radius_bins
        )

        # compute 2point correlation function : (z_obs, l_obs, l_obs, radius)
        _clustering_zbin_lbin_rbin_buf = (
            self.cluster_statitstics_modeling.integrate_quantity_in_k_space(
                window_zbin_lbin_k[:, np.newaxis, np.newaxis, :, :]
                * pk_zbin_lbin_lbin_k[:, :, :, np.newaxis, :]
            )
        )

        # xi(l_obs_i, l_obs_j) = xi(l_obs_j, l_obs_i) so we reshape
        # and keep only one of them, with a (z_obs, l_obs, radius) output
        triangle_indexes = np.triu_indices(len(lambda_obs_bins) - 1)
        clustering_zbin_lbin_rbin = _clustering_zbin_lbin_rbin_buf[
            :, triangle_indexes[0], triangle_indexes[1], :
        ]

        if not return_intermediate_products:
            return clustering_zbin_lbin_rbin

        intermediate_products_zbin_lbin = {
            "pk_zbin_lbin_lbin_k": pk_zbin_lbin_lbin_k,
            "window_zbin_lbin_k": window_zbin_lbin_k,
            "vol_zbin_rbin": vol_zbin_rbin,
            "nc_zbin_lbin": nc_zbin_lbin,
            "dvdz_zbin_lbin_z": dvdz_zbin_lbin_z,
        }
        return clustering_zbin_lbin_rbin, intermediate_products_zbin_lbin

    # ----------------------
    # clustering covariance
    # ----------------------

    def compute_cov(
        self,
        pk_zbin_lbin_lbin_k,
        window_zbin_lbin_k,
        vol_zbin_rbin,
        dvdz_zbin_lbin_z,
        nc_zbin_lbin,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        pk_zbin_lbin_lbin_k : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        window_zbin_lbin_k : numpy.ndarray
            Cluster count covariance window (z_obs, l_obs, k),
            with (k) in cluster_statitstics_modeling.kenel_tables.
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        vol_zbin_rbin : numpy.ndarray
            Spherical shell volume (z_obs, radius).
            Is in the intermediate_products_zbin_lbin output of compute_binned_clustering.
        dvdz_zbin_lbin_z : numpy.ndarray
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
        _, radius_bins_size = vol_zbin_rbin.shape

        ########################################
        # Cluster statistics modeling quantities
        ########################################

        # Compute observed volume in each redshift bin : (z_obs, l_obs)
        vol_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            dvdz_zbin_lbin_z
        )
        # Compute output shot-noise terms : (z_obs, l_obs, l_obs)
        vol_over_nc_zbin_lbin_lbin = (
            vol_zbin_lbin[:, :, np.newaxis]
            / nc_zbin_lbin[:, :, np.newaxis]
            * np.identity(lambda_obs_bins_size)[np.newaxis, :, :]
        )

        #############################
        # Compute nuisance parameters
        #############################

        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0, beta=1, gamma=0 (see Euclid Collaboration :
        #    Fumagalli et al. 2022)
        alpha = np.zeros((z_obs_bins_size, lambda_obs_bins_size))
        beta = np.ones((z_obs_bins_size, lambda_obs_bins_size))
        gamma = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        # Combine alpha, beta with pk, vol and reshape to be used
        # in cov_g, cov_ng integral
        beta_pk_zbin_lbin_lbin_k = (
            beta[:, :, np.newaxis, np.newaxis]
            * beta[:, np.newaxis, :, np.newaxis]
            * pk_zbin_lbin_lbin_k
        )
        avol_bpk_zbin_lbin_lbin_k = (
            # reshape alpha to be (z_obs, l_obs, l_obs)
            (1 + alpha)[:, :, np.newaxis, np.newaxis]
            * (1 + alpha)[:, np.newaxis, :, np.newaxis]
            * vol_over_nc_zbin_lbin_lbin[:, :, :, np.newaxis]
            + beta_pk_zbin_lbin_lbin_k
        )

        ####################
        # Compute covariance
        ####################

        # define cluster clustering bin numbers for loops
        z_bin_loop = range(z_obs_bins_size)
        lambda_bin_loop = range(lambda_obs_bins_size)
        rad_bin_loop = range(radius_bins_size)

        # cov_g, cov_ng are TWO TERMS OF EQ. 73
        _cov_g_zbin_4lbin_2rbin = np.zeros(
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
        _cov_ng_zbin_4lbin_2rbin = np.zeros(
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

        # note : this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_loop:
            for ind_lambda_j in lambda_bin_loop:
                for ind_radius in rad_bin_loop:
                    _cov_ng_zbin_4lbin_2rbin[
                        :,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_radius,
                        ind_radius,
                    ] = (
                        self.cluster_statitstics_modeling.integrate_quantity_in_k_space(
                            window_zbin_lbin_k[:, ind_radius, :]
                            * beta_pk_zbin_lbin_lbin_k[
                                :, ind_lambda_i, ind_lambda_j, :
                            ],
                        )
                        * (1 + gamma[:, ind_lambda_i])
                        * vol_over_nc_zbin_lbin_lbin[:, ind_lambda_i, ind_lambda_i]
                        * (1 + gamma[:, ind_lambda_j])
                        * vol_over_nc_zbin_lbin_lbin[:, ind_lambda_j, ind_lambda_j]
                        / vol_zbin_rbin[:, ind_radius]
                    )

                for ind_lambda_k in lambda_bin_loop:
                    for ind_lambda_h in lambda_bin_loop:

                        # somehow using _integrate_quantity_in_k is much faster then
                        # integrate_quantity_in_k_space here, to be investigated

                        # gaussian term
                        _cov_g_zbin_4lbin_2rbin[
                            :,
                            ind_lambda_i,
                            ind_lambda_j,
                            ind_lambda_k,
                            ind_lambda_h,
                            :,
                            :,
                        ] = self.cluster_statitstics_modeling._integrate_quantity_in_k(
                            window_zbin_lbin_k[:, np.newaxis, :, :]
                            * window_zbin_lbin_k[:, :, np.newaxis, :]
                            * avol_bpk_zbin_lbin_lbin_k[
                                :,
                                ind_lambda_i,
                                ind_lambda_k,
                                np.newaxis,
                                np.newaxis,
                                :,
                            ]
                            * avol_bpk_zbin_lbin_lbin_k[
                                :,
                                ind_lambda_j,
                                ind_lambda_h,
                                np.newaxis,
                                np.newaxis,
                                :,
                            ]
                            * self.cluster_statitstics_modeling.kernel_tables["dk"],
                        )

        # Compute the covariance
        _cov_clustering_zbin_4lbin_2rbin = (
            (_cov_g_zbin_4lbin_2rbin + _cov_ng_zbin_4lbin_2rbin)
            + (_cov_g_zbin_4lbin_2rbin + _cov_ng_zbin_4lbin_2rbin).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_clustering bins
            )
        ) / vol_zbin_lbin[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(l_obs_i, l_obs_j, l_obs_k, l_obs_l) = cov_xi(l_obs_j, l_obs_i, l_obs_l, l_obs_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_bins_size)
        # simplify first pair
        _cov_clustering_zbin_3lbin_2rbin = _cov_clustering_zbin_4lbin_2rbin[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        _cov_clustering_zbin_2lbin_2rbin = _cov_clustering_zbin_3lbin_2rbin[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (z_obs, l_obs, l_obs, radius, radius),
        # make it (z_obs, z_obs, l_obs, l_obs, radius, radius),
        # being diagonal in (z_obs, z_obs)
        # ---------------------------------------------------------------------------------
        # OBS: for simplicity and homeneity with other outputs, it will be written down as
        # cov_clustering_zbin_lbin_rbin instead of cov_clustering_2zbin_2lbin_2rbin
        # ---------------------------------------------------------------------------------
        cov_clustering_zbin_lbin_rbin = (
            np.identity(z_obs_bins_size)[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * _cov_clustering_zbin_2lbin_2rbin
        )
        return cov_clustering_zbin_lbin_rbin

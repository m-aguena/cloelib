# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.profile import Profile
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Cluster profile lensing

"""


class ClusterWeakLensing:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        profile: Profile,
        halo_concentration: float,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        profile : Profile
            Halo weak lensing radial profile object
        halo_concentration : float
            Halo concentration
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

        # hardcoded quantities for integration
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

    def compute_binned_deltasigma(self, z_obs_bins, lambda_obs_bins, radius_bins):
        """Compute excess surface density.

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
        deltasigma_zbin_lbin_rbin : numpy.ndarray
            Excess surface density in redshift, richness, and radial bins.
        """
        z_obs_bins_size = len(z_obs_bins) - 1

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # volume element in each redshift and richness bin shape : (z_obs, l_obs, z)
        dvdz_zbin_lbin_z = (
            self.cluster_statitstics_modeling.compute_binned_volume_element(
                z_obs_bins, lambda_obs_bins, self.z_tab_sig
            )
        )
        # P(lambda_obs_bins|M, z) : (l_obs, M, z)
        p_lbin_z_m = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        # integral of P(lambda_obs_bins|M, z)*dn/dM on mass : (l_obs, z)
        _p_lbin_z = self.cluster_statitstics_modeling.integrate_kernel_in_mass_w_hmf(
            np.ones((1, 1)), p_lbin_z_m
        )
        # cluster counts : (z_obs, l_obs)
        nc_zbin_lbin = (
            self.cluster_statitstics_modeling.integrate_lbin_kernel_in_true_volume(
                _p_lbin_z, dvdz_zbin_lbin_z
            )
        )

        ########################
        # Compute binned profile
        ########################

        # mass/richness part

        # excess surface mass density
        deltasigma_z_m_rbin = self.profile.excess_surface_mass_density(
            radius_bins[:-1],
            self.cluster_statitstics_modeling.kernel_tables["ztrue"],
            self.cluster_statitstics_modeling.kernel_tables["M"],
            self.halo_concentration,
        )
        # excess surface mass density in a richness bin, integrated on mass w HMF : (l_obs, z)
        deltasigma_lbin_z_rbin = (
            self.cluster_statitstics_modeling.integrate_kernel_in_mass_w_hmf(
                deltasigma_z_m_rbin, p_lbin_z_m
            )
        )

        # redshift part

        # Effective inverse critical surface mass density.
        inv_sig_crit_eff_zbin_z = np.zeros(
            (
                z_obs_bins_size,
                self.cluster_statitstics_modeling.kernel_tables["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_bins_size):
            inv_sig_crit_eff_zbin_z[ind_z] = self.profile.m_sig_crit_m1(
                self.cluster_statitstics_modeling.kernel_tables["ztrue"],
                ind_z,
            )

        deltasigma_zbin_lbin_rbin = (
            self.cluster_statitstics_modeling.integrate_lbin_kernel_in_true_volume(
                deltasigma_lbin_z_rbin,
                dvdz_zbin_lbin_z * inv_sig_crit_eff_zbin_z[:, np.newaxis, :],
            )
        ) / nc_zbin_lbin[:, :, np.newaxis]
        return deltasigma_zbin_lbin_rbin

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
        deltasigma_mean_values : numpy.ndarray
            Excess surface density in redshift, richness, and radial bins.
        """
        z_obs_bins_size = len(z_obs_bins) - 1

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # P(z_obs_bin|lambda_obs, ztrue) : (z_obs, lambda_obs, ztrue)
        prob_z_obs_bins = (
            self.cluster_statitstics_modeling.compute_binned_redshift_obs_probability(
                z_obs_bins, lambda_obs_bins, self.z_tab_sig
            )
        )
        # P(lambda_obs_bins|M, z) : (lambda_obs, M, ztrue)
        prob_lambda_obs_bins = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_in_true_redshift(
            # integral of P(lambda_obs_bins|M, z)*dn/dM on mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_in_mass(
                np.ones((1, 1)), prob_lambda_obs_bins
            ),
            prob_z_obs_bins,
        )

        ########################
        # Compute binned profile
        ########################

        # mass/richness part

        # excess surface mass density : (ztrue, M, radius)
        deltasigma = self.profile.excess_surface_mass_density(
            radius_bins[:-1],
            self.cluster_statitstics_modeling.kernel_tables["ztrue"],
            self.cluster_statitstics_modeling.kernel_tables["M"],
            self.halo_concentration,
        )
        # excess surface mass density in a richness bin, integrated on mass w HMF : (lambda_obs, ztrue, M, radius)
        deltasigma_lambda_obs_bins = (
            self.cluster_statitstics_modeling.integrate_in_mass(
                deltasigma, prob_lambda_obs_bins
            )
        )

        # redshift part

        # Effective inverse critical surface mass density : (z_obs, ztrue)
        inv_sig_crit_eff = np.zeros(
            (
                z_obs_bins_size,
                self.cluster_statitstics_modeling.kernel_tables["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_bins_size):
            inv_sig_crit_eff[ind_z] = self.profile.m_sig_crit_m1(
                self.cluster_statitstics_modeling.kernel_tables["ztrue"],
                ind_z,
            )

        # output : (z_obs, lambda_obs, radius)
        deltasigma_mean_values = (
            self.cluster_statitstics_modeling.integrate_in_true_redshift(
                deltasigma_lambda_obs_bins,
                prob_z_obs_bins * inv_sig_crit_eff[:, np.newaxis, :],
            )
        ) / cluster_counts[:, :, np.newaxis]
        return deltasigma_mean_values

# General imports
import numpy as np

from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance

# cloelib imports
from cloelib.observables.clusters.halo_profile import HaloProfile
from cloelib.observables.clusters.selection_function import SelectionFunction
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
        profile: HaloProfile,
        halo_concentration: float,
        selection_function: SelectionFunction,
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
        selection_function : SelectionFunction
            Selection function object
        """
        halo_abundance = cluster_statitstics_modeling.halo_abundance
        Delta_abundance = halo_abundance.overdensity_type
        Delta_profile = profile.core.overdensity_type
        if isinstance(halo_abundance, CastroHaloAbundance) and Delta_profile != "vir":
            raise ValueError(
                f"If the Castro HMF is used, only virial overdensities can be considered. The current overdensity in the profile modeling is {Delta_profile}."
            )
        if Delta_abundance != Delta_profile:
            raise ValueError(
                f"The overdensity definition of the mass profile ({Delta_profiles}) differs from the one adopted for halo abundance modeling ({Delta_abundance}).)"
            )

        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.profile = profile
        self.selection_function = selection_function

        # internal values
        self.halo_concentration = halo_concentration

    def _get_profile(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        effective_inverse_critical_surface_mass_density=None,
    ):
        """Compute weak lensing profile, it can be excess surface density or reduced shear.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.
        effective_inverse_critical_surface_mass_density : numpy.array, None
            The effective inverse of the critical surface density.
            If provided, it must be shape (ztrue, M, radius) and this function
            returns the reduced shear, else it returns the excess surface density.

        Returns
        -------
        wl_profile_mean_values : numpy.ndarray
            Weak lensing quantity (excess surface density or reduced shear) in redshift,
            richness, and radial bins.
        """
        ############################################
        # Get cluster statistics modeling quantities
        ############################################
        # P(lambda_obs_bin,z_obs_bin|M, z): (z_obs_bin,lambda_obs_bin, ztrue, mass)
        window_redshift_lambda_obs = (
            self.cluster_statitstics_modeling.window_redshift_richness_observed(
                self.selection_function, z_obs_edges, lambda_obs_edges
            )
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs_bin,z_obs_bin|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_redshift_lambda_obs
            ),
        )

        ########################
        # Compute binned profile
        ########################

        # mass/richness part

        # excess surface mass density : (ztrue, M, radius)
        excess_surface_mass_density = self.profile.excess_surface_mass_density(
            radius_edges[:-1],
            self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
            self.cluster_statitstics_modeling.tabulated_integrands["M"],
            self.halo_concentration,
        )
        # excess surface mass density in a richness bin, integrated on mass w HMF
        # Dimension: (z_obs_bin, lambda_obs_bin, ztrue, radius)
        excess_surface_density_in_window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                excess_surface_mass_density,
                window_redshift_lambda_obs[:, :, :, :, np.newaxis],
            )
        )

        # redshift part

        # Apply effective inverse critical surface mass density if provided
        if effective_inverse_critical_surface_mass_density is not None:
            excess_surface_density_in_window_lambda_obs_mass_integrated *= (
                effective_inverse_critical_surface_mass_density[
                    :, np.newaxis, :, np.newaxis
                ]
            )

        # output : (z_obs, lambda_obs, radius)
        wl_profile_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                excess_surface_density_in_window_lambda_obs_mass_integrated,
            )
        ) / cluster_counts[:, :, np.newaxis]
        return wl_profile_mean_values

    def get_DeltaSigma(self, z_obs_edges, lambda_obs_edges, radius_edges):
        """Compute excess surface density profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.

        Returns
        -------
        deltasigma_mean_values : numpy.ndarray
            Excess surface density in redshift, richness, and radial bins.
        """
        # output : (z_obs, lambda_obs, radius)
        return self._get_profile(
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            effective_inverse_critical_surface_mass_density=None,
        )

    def get_gt(self, z_obs_edges, lambda_obs_edges, radius_edges):
        """Compute reduced shear profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.

        Returns
        -------
        gt_mean_values : numpy.ndarray
           Reduced shear in redshift, richness, and radial bins.
        """
        z_obs_edges_size = len(z_obs_edges) - 1

        # Effective inverse critical surface mass density : (z_obs, ztrue)
        effective_inverse_critical_surface_mass_density = np.zeros(
            (
                z_obs_edges_size,
                self.cluster_statitstics_modeling.tabulated_integrands["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_edges_size):
            effective_inverse_critical_surface_mass_density[ind_z] = (
                self.profile.core.sigma_crit_inv_eff(
                    self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
                    ind_z,
                )
            )

        # output : (z_obs, lambda_obs, radius)
        return self._get_profile(
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            effective_inverse_critical_surface_mass_density,
        )

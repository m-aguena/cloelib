"""

## Notes :

- Cluster profile lensing

"""

# General imports
import numpy as np


# cloelib imports
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_profile import HaloProfile
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax


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
                f"The overdensity definition of the mass profile ({Delta_profile}) differs from the one adopted for halo abundance modeling ({Delta_abundance}).)"
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

        # integral of P(z_obs|lambda_obs, z) on z_obs bins : (z_obs, lambda_obs, ztrue)
        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            self.selection_function, z_obs_edges, lambda_obs_edges
        )
        # integral of P(lambda_obs|M, z) on lambda_obs bins : (lambda_obs, M, ztrue)
        window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            self.selection_function, lambda_obs_edges
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_lambda_obs
            ),
            window_z_obs,
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
        # excess surface mass density in a richness bin, integrated on mass w HMF : (lambda_obs, ztrue, radius)
        excess_surface_density_in_window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                excess_surface_mass_density, window_lambda_obs
            )
        )

        # redshift part

        # Apply effective inverse critical surface mass density if provided
        window_z_obs_use = window_z_obs
        if effective_inverse_critical_surface_mass_density is not None:
            window_z_obs_use *= effective_inverse_critical_surface_mass_density[
                :, np.newaxis, :
            ]

        # output : (z_obs, lambda_obs, radius)
        wl_profile_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                excess_surface_density_in_window_lambda_obs_mass_integrated,
                window_z_obs_use,
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

    def get_gt(
        self, z_obs_edges, lambda_obs_edges, radius_edges, opt_sel_bias_params=None
    ):
        """Compute reduced shear profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.
        opt_sel_bias_params: tuple, None
            If not None, applies the optical selection bias correction to the
            profile multiplying it by
            ``self.optical_selection_bias_correction(radius_edges, *opt_sel_bias_params)``.
            The values must be `opt_sel_bias_params=(R0, A, alpha, beta, gamma)``,
            where each individual parameter must be either float or have shape
            (redshift, richness, radius) bins.

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
        opt_sel_corr = 1
        if opt_sel_bias_params is not None:
            opt_sel_corr = self.optical_selection_bias_correction(
                radius_edges, *opt_sel_bias_params
            )
        return (
            self._get_profile(
                z_obs_edges,
                lambda_obs_edges,
                radius_edges,
                effective_inverse_critical_surface_mass_density,
            )
            * opt_sel_corr
        )

    @staticmethod
    def optical_selection_bias_correction(R, R0, A, alpha, beta, gamma):
        """
        Correction for the weak lensing optical selection bias to account for
        miscentering and projection effects. To be multiplied directly to the WL
        profile integrated in observed richness and redshift. Effect measured in
        Ingrao et al. 2026 (https://doi.org/10.48550/arXiv.2605.02723).

        Parameters
        ----------
        R: numpy.ndarray
            Radius of the profile in Mpc
        R0: numpy.ndarray
            Transition scale in Mpc, dimensions should be (z_obs_bins, lambda_obs_bins)
        A: numpy.ndarray
            Amplitude of the correction, dimensions should be (z_obs_bins, lambda_obs_bins)
        alpha: numpy.ndarray
            Slope at small radii, dimensions should be (z_obs_bins, lambda_obs_bins)
        beta: numpy.ndarray
            Slope at large radii, dimensions should be (z_obs_bins, lambda_obs_bins)
        gamma: numpy.ndarray
            Smoothness of the transition between slopes, dimensions should be (z_obs_bins, lambda_obs_bins)


        Retruns
        -------
            Correction for WL optical selection bias. Dimension (z_obs_bins, lambda_obs_bins)

        Note
        ----
        Reasonable values for the parameters are: R0=1.20cMpc/h, A=0.20, alpha=4.0,
        beta=−0.3 , gamma=1.6
        """
        return (
            A * (R / R0) ** alpha * (1 + (R / R0**gamma)) ** ((alpha - beta) / gamma)
            + 1
        )

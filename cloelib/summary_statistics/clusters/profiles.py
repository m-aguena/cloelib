# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.profile import Profile
from cloelib.summary_statistics.clusters.counts import ClusterCounts

# import jax

"""

## Notes :

- Cluster profile lensing

"""


class ClusterWeakLensing:
    def __init__(
        self,
        cluster_counts: ClusterCounts,
        profile: Profile,
        halo_concentration: float,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_counts : ClusterCounts
            Cluster counts summary statistics object
        profile : Profile
            Halo weak lensing radial profile object
        halo_concentration : float
            Halo concentration
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_counts = cluster_counts

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

    def compute_binned_quantities(self, z_obs_bins, lambda_obs_bins, radius_bins):
        """compute reduced shear.

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
        gt_zbin_lbin_rbin : numpy.ndarray
            Reduced shear in redshift, richness, and radial bins.
        """
        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        z_obs_bins_size = len(z_obs_bins) - 1
        radius_bins_size = len(radius_bins) - 1

        gt_zbin_lbin_rbin = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, radius_bins_size)
        )

        # get cluster counts quantities
        nc_zbin_lbin, counts_aux = self.cluster_counts.compute_binned_quantities(
            z_obs_bins=z_obs_bins,
            lambda_obs_bins=lambda_obs_bins,
        )

        # compute profile quantities
        for ind_radius in range(radius_bins_size):
            excess_surface_mass_density = self.profile.excess_surface_mass_density(
                np.atleast_1d(radius_bins[ind_radius]),
                self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                self.cluster_counts.cluster_stat_kernel_tables["mass"],
                self.halo_concentration,
            )
            for ind_lambda in range(lambda_obs_bins_size):
                excesssurfacemassdensity = simps(
                    counts_aux["Plob_M_z"][ind_lambda]
                    * self.cluster_counts.cluster_stat_kernel_tables["dndm_z"]
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.cluster_counts.cluster_stat_kernel_tables["mass"],
                    axis=1,
                )

                for ind_z in range(z_obs_bins_size):
                    gt_zbin_lbin_rbin[ind_z, ind_lambda, ind_radius] = (
                        (1.0)
                        / nc_zbin_lbin[ind_z, ind_lambda]
                        * simps(
                            self.profile.m_sig_crit_m1(
                                self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                                ind_z,
                            )
                            * counts_aux["dV_dzob"][ind_z, ind_lambda]
                            * excesssurfacemassdensity,
                            x=self.cluster_counts.cluster_stat_kernel_tables["ztrue"],
                        )
                    )
        return gt_zbin_lbin_rbin

# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.profile import Profile
from cloelib.summary_statistics.clusters import ClusterCounts

# import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""


class ClusterWL:
    def __init__(
        self,
        cluster_counts_statistics: ClusterCounts,
        profile: Profile,
        halo_concentration: float,
    ):
        # cluster counts statistics, for integration tables
        self.cluster_counts_statistics = cluster_counts_statistics

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

    def compute_binned_properties(
        self, z_obs_bins, lambda_obs_bins, radius_bins, nc_zbin_lbin, Plob_M_z, dV_dzob
    ):
        """compute reduced shear.
        returns reduced shear array

        Parameters
        ----------
        nc_zbin_lbin : numpy.ndarray
            cluster number counts
        Plob_M_z: numpy.ndarray
            probablity of observed richness given mass and redshift
        dV_dzob: numpy.ndarray
            volume element

        Returns
        -------
        numpy.ndarray
            reduced shear array
        """
        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        z_obs_bins_size = len(z_obs_bins) - 1
        radius_bins_size = len(radius_bins) - 1

        gt_zbin_lbin_rbin = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, radius_bins_size)
        )

        for ind_radius in range(radius_bins_size):
            excess_surface_mass_density = self.profile.excess_surface_mass_density(
                np.atleast_1d(radius_bins[ind_radius]),
                self.cluster_counts_statistics.integ_ztrue_arr,
                self.cluster_counts_statistics.integ_mass_arr,
                self.halo_concentration,
            )
            for ind_lambda in range(lambda_obs_bins_size):
                excesssurfacemassdensity = simps(
                    Plob_M_z[ind_lambda]
                    * self.cluster_counts_statistics.dndm_z
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.cluster_counts_statistics.integ_mass_arr,
                    axis=1,
                )

                for ind_z in range(z_obs_bins_size):
                    gt_zbin_lbin_rbin[ind_z, ind_lambda, ind_radius] = (
                        (1.0)
                        / nc_zbin_lbin[ind_z, ind_lambda]
                        * simps(
                            self.profile.m_sig_crit_m1(
                                self.cluster_counts_statistics.integ_ztrue_arr, ind_z
                            )
                            * dV_dzob[ind_z, ind_lambda]
                            * excesssurfacemassdensity,
                            x=self.cluster_counts_statistics.integ_ztrue_arr,
                        )
                    )
        return gt_zbin_lbin_rbin

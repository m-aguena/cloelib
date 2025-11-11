# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.profile import Profile
from cloelib.summary_statistics.clusters.counts import ClusterCounts

# import jax

"""

## Notes:

- Cluster profile lensing

"""


class ClusterWL:
    def __init__(
        self,
        integ_tables: dict,
        profile: Profile,
        halo_concentration: float,
    ):
        # integration tables
        self.integ_tables = integ_tables

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

    def compute_binned_quantities(
        self, z_obs_bins, lambda_obs_bins, radius_bins, nc_zbin_lbin, Plob_M_z, dV_dzob
    ):
        """compute reduced shear.

        Parameters
        ----------
        z_obs_bins: numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins: numpy.ndarray
            Richness bins for the integration.
        radius_obs_bins: numpy.ndarray
            Radial bins for the profile.
        nc_zbin_lbin : numpy.ndarray
            Number counts in richness and redshift bins
        Plob_M_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
        dV_dzob : numpy.ndarray
            Observed volume element (dV/dz_ob) in each redshift and richness bin

        Returns
        -------
        gt_zbin_lbin_rbin: numpy.ndarray
            Reduced shear in redshift, richness, and radial bins.
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
                self.integ_tables["ztrue"],
                self.integ_tables["mass"],
                self.halo_concentration,
            )
            for ind_lambda in range(lambda_obs_bins_size):
                excesssurfacemassdensity = simps(
                    Plob_M_z[ind_lambda]
                    * self.integ_tables["dndm_z"]
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.integ_tables["mass"],
                    axis=1,
                )

                for ind_z in range(z_obs_bins_size):
                    gt_zbin_lbin_rbin[ind_z, ind_lambda, ind_radius] = (
                        (1.0)
                        / nc_zbin_lbin[ind_z, ind_lambda]
                        * simps(
                            self.profile.m_sig_crit_m1(
                                self.integ_tables["ztrue"], ind_z
                            )
                            * dV_dzob[ind_z, ind_lambda]
                            * excesssurfacemassdensity,
                            x=self.integ_tables["ztrue"],
                        )
                    )
        return gt_zbin_lbin_rbin

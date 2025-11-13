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

    def compute_binned_profile(self, z_obs_bins, lambda_obs_bins, radius_bins):
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

        # output
        gt_zbin_lbin_rbin = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, radius_bins_size)
        )

        # get cluster statistics modeling quantities
        dv_dzob = self.cluster_statitstics_modeling.compute_binned_volume(
            z_obs_bins, lambda_obs_bins, self.z_tab_sig
        )
        p_lbin_M_z = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        _p_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                p_lbin_M_z
            )
        )  # integral of Plob_M_z*dndm_z on mass.

        # cluster counts
        nc_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            _p_lbin_z[np.newaxis, :] * dv_dzob
        )

        # pre-compute excess surface mass density
        excess_surface_mass_density = self.profile.excess_surface_mass_density(
            radius_bins,
            self.cluster_statitstics_modeling.kernel_tables["ztrue"],
            self.cluster_statitstics_modeling.kernel_tables["mass"],
            self.halo_concentration,
        )
        # pre-compute effective inverse critical surface mass density.
        # def m_sig_crit_m1(self, z, zbin):
        m_sig_crit_m1 = np.zeros(
            (
                z_obs_bins_size,
                self.cluster_statitstics_modeling.kernel_tables["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_bins_size):
            m_sig_crit_m1[ind_z] = self.profile.m_sig_crit_m1(
                self.cluster_statitstics_modeling.kernel_tables["ztrue"],
                ind_z,
            )

        # compute profile quantities
        for ind_radius in range(radius_bins_size):
            for ind_lambda in range(lambda_obs_bins_size):
                gt_lbdobs_z = simps(
                    p_lbin_M_z[ind_lambda]
                    * self.cluster_statitstics_modeling.kernel_tables["dndm_z"]
                    * excess_surface_mass_density[:, :, ind_radius],
                    x=self.cluster_statitstics_modeling.kernel_tables["mass"],
                    axis=1,
                )
                for ind_z in range(z_obs_bins_size):
                    gt_zbin_lbin_rbin[ind_z, ind_lambda, ind_radius] = (
                        simps(
                            m_sig_crit_m1[ind_z]
                            * dv_dzob[ind_z, ind_lambda]
                            * gt_lbdobs_z,
                            x=self.cluster_statitstics_modeling.kernel_tables["ztrue"],
                        )
                        / nc_zbin_lbin[ind_z, ind_lambda]
                    )
        return gt_zbin_lbin_rbin

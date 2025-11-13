# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Cluster counts

"""


class ClusterCounts:
    """Object to compute cluster counts

    Attributes
    ----------
    l_m_tab_sig : list
        Number of points to be used for the lambda_obs integration
        in each lambda_obs bin. Must be same size of lambda_obs_bins.
    z_tab_sig : int
        Number of points to be used for z_obs integration.
    """

    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        covariance: HaloCovariance,
        photoz_rsd_correction,
    ):
        """
        Initializes the cluster counts

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        covariance : HaloCovariance
            Halo covariance object
        photoz_rsd_correction : function
            Function that computers the RSD correction
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.covariance = covariance

        # ---------------------------------------------------------------
        # Note : This is a patch as this function is currently implemented
        # in HaloClustering, should it be moved to SelectionFunction?
        # ---------------------------------------------------------------
        self.photoz_rsd_correction = photoz_rsd_correction

        # hardcoded quantities for integration
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

    def compute_binned_counts(
        self,
        z_obs_bins,
        lambda_obs_bins,
        return_intermediate_products=True,
    ):
        """Computes binned quantities (counts+aux).

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.
        return_intermediate_products : bool
            If true, also returns `intermediate_products_zbin_lbin`, a dictionary
            with the intermediate products computed.

        Returns
        -------
        nc_zbin_lbin : numpy.ndarray
            Number counts in redshift and richness bins
        intermediate_products_zbin_lbin (optional) : dict
            Dictionary with intermidate products that can be used for other computations.
            Returned only when `return_intermediate_products` is true.
            Contains :

                * Plob_M_z (numpy.ndarray) : Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
                * dV_dzob (numpy.ndarray) : Observed volume element (dV/dz_ob) in each redshift and richness bin
                * nc_lbdobs_z (numpy.ndarry) : integral of Plob_M_z*dndm_z on mass.
        """
        dv_dzob = self.cluster_statitstics_modeling.compute_binned_volume(
            z_obs_bins, lambda_obs_bins, self.z_tab_sig
        )
        p_lbin_M_z = (
            self.cluster_statitstics_modeling.compute_binned_lambda_obs_probability(
                lambda_obs_bins, self.l_m_tab_sig
            )
        )
        p_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                p_lbin_M_z
            )
        )  # integral of Plob_M_z*dndm_z on mass.

        nc_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            p_lbin_z[np.newaxis, :] * dv_dzob
        )

        if not return_intermediate_products:
            return nc_zbin_lbin

        return nc_zbin_lbin, {
            "Plob_M_z": p_lbin_M_z,
            "dV_dzob": dv_dzob,
            "nc_lbdobs_z": p_lbin_z,
        }

    # -------------------
    # cluster counts cov
    # -------------------

    def _compute_spatial_cov(self, z_obs_bins):
        """Computes only spatial part of the covariance.

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.

        Returns
        -------
        spatial_cov : numpy.ndarray
            Spatial part of the covariance
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        z_mid = 0.5 * (z_obs_bins[1:] + z_obs_bins[:-1])

        # initialization
        spatial_cov = np.zeros((z_obs_bins_size, z_obs_bins_size))
        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()
        # self.rint = np.zeros((z_obs_bins_size,len(self.cluster_statitstics_modeling.kernel_tables["k"]),L+1))

        # power spectrum at the center of observed redshift bins
        pk = self.cluster_statitstics_modeling.halo_statistics.matter_power_spectrum(
            z_mid, self.cluster_statitstics_modeling.kernel_tables["k"]
        )

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        photoz_corr0 = self.photoz_rsd_correction(z_mid, 0)[
            0
        ]  # can neglect richness dependence here
        pk *= photoz_corr0

        # richness-independent quantites that can be computed outside the lambda loop
        for ind_z in range(z_obs_bins_size):

            z_tab = np.linspace(
                z_obs_bins[ind_z], z_obs_bins[ind_z + 1], self.z_tab_sig
            )

            spatial_cov[ind_z, : (ind_z + 1)] = (
                1
                / (2 * np.pi**2)
                * simps(
                    (
                        self.cluster_statitstics_modeling.kernel_tables["k"] ** 2
                        * np.sqrt(pk[ind_z] * pk[: (ind_z + 1)])
                    )
                    * self.covariance.cov_window(ind_z, z_tab, KL),
                    x=self.cluster_statitstics_modeling.kernel_tables["k"],
                    axis=-1,
                )
            )
            spatial_cov[: (ind_z + 1), ind_z] = spatial_cov[ind_z, : (ind_z + 1)]

        return spatial_cov

    def compute_cov(self, z_obs_bins, nc_zbin_lbin, Plob_M_z, dV_dzob):
        """Computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        nc_zbin_lbin : numpy.ndarray
            Number counts in redshift and richness bins
        Plob_M_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z),
            with masses and redshifts being the values in cluster_statitstics_modeling.kernel_tables.
            Is in the intermediate_products_zbin_lbin output of compute_binned_counts.
        dV_dzob : numpy.ndarray
            Observed volume element (dV/dz_ob) in each redshift and richness bin.
            Is in the intermediate_products_zbin_lbin output of compute_binned_counts.

        Returns
        -------
        cov_nc_zbin_lbin : numpy.ndarray
            Covariance number counts in redshift and richness bins
        """

        # halo bias
        _b_lbin_z = (
            self.cluster_statitstics_modeling.integrate_binned_quantity_in_mass_w_hmf(
                Plob_M_z * self.cluster_statitstics_modeling.kernel_tables["bias_z"]
            )
        )  # integral of Plob_M_z*dndm_z*bias_z on mass.
        hbias_zbin_lbin = self.cluster_statitstics_modeling.integrate_2d_binned_quantity_in_true_redshift(
            _b_lbin_z[np.newaxis, :] * dV_dzob
        )

        # spatial component of covariance
        spatial_cov = self._compute_spatial_cov(z_obs_bins)

        # shot noise
        _shot_noise = (
            np.diag(nc_zbin_lbin.flatten())
            .reshape(*nc_zbin_lbin.shape, *nc_zbin_lbin.shape)
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance
        cov_nc_zbin_lbin = _shot_noise + (
            hbias_zbin_lbin[np.newaxis, :, np.newaxis, :]
            * hbias_zbin_lbin[:, np.newaxis, :, np.newaxis]
            * spatial_cov[:, :, np.newaxis, np.newaxis]
        )

        return cov_nc_zbin_lbin

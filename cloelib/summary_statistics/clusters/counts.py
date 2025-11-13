# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.hmf_bias import HMFBias
from cloelib.observables.clusters.selection_function import SelectionFunction

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
    z_tab_sig : float
    cluster_stat_kernel_tables : dict
        Dictionary with tables that will be used for integrations. Contains :

            * k (np.ndarray) : Values of k to be used in integrations
            * mass (np.ndarray) : Values of mass to be used in integrations
            * lambda_true (np.ndarray) : Values of true richness to be used in integrations
            * ztrue (np.ndarray) : Values of true redshift to be used in integrations
            * Pltrue_M_z(np.ndarray) : Values for P(lambda_true|mass, z)
            * dvdzdomega_z1z2(np.ndarray) : Values for volume element at each redshift
            * dndm_z(np.ndarray) : Values for the halo mass function dn/dmdz(mass, z)
            * bias_z(np.ndarray) : Values for the halo bias halo_bias(mass, z)
    """

    def __init__(
        self,
        hmfbias: HMFBias,
        selectionfunction: SelectionFunction,
        covariance: HaloCovariance,
        photoz_rsd_correction,
        integ_k_arr: np.ndarray,
        integ_mass_arr: np.ndarray,
        integ_lambda_true_arr: np.ndarray,
        integ_ztrue_arr: np.ndarray,
        area: float = 10313,
    ):
        """
        Initializes the cluster counts

        Parameters
        ----------
        hmfbias : HMFBias
            Halo mass function and bias object
        selectionfunction : SelectionFunction
            Selection function object
        covariance : HaloCovariance
            Halo covariance object
        photoz_rsd_correction : function
            Function that computers the RSD correction
        integ_k_arr : np.ndarray
            Values of k to be used in integrations, stored in cluster_stat_kernel_tables
        integ_mass_arr : np.ndarray
            Values of mass to be used in integrations, stored in cluster_stat_kernel_tables
        integ_lambda_true_arr : np.ndarray
            Values of true richness to be used in integrations, stored in cluster_stat_kernel_tables
        integ_ztrue_arr : np.ndarray
            Values of true redshift to be used in integrations, stored in cluster_stat_kernel_tables
        area : float
            Area of the survey in deg2.

        """
        # observable objects
        self.hmfbias = hmfbias
        self.halo_statistics = self.hmfbias.halo_statistics
        self.selectionfunction = selectionfunction
        self.covariance = covariance

        # ---------------------------------------------------------------
        # Note : This is a patch as this function is currently implemented
        # in HaloClustering, should it be moved to SelectionFunction?
        # ---------------------------------------------------------------
        self.photoz_rsd_correction = photoz_rsd_correction

        # internal values
        self.area = area

        ################### QUANTITIES FOR INTEGRATION ###################

        # hardcoded quantities
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        # integration tables
        self.cluster_stat_kernel_tables = {
            "k": integ_k_arr,  # k array
            "mass": integ_mass_arr,  # mass array in Msun h^-1
            "lambda_true": integ_lambda_true_arr,  # true richness array
            "ztrue": integ_ztrue_arr,  # true redshift array
            # P(ltr|M,ztr), this quantity is also used by cluster clustering
            "Pltrue_M_z": self.selectionfunction.P_lnlbd(
                integ_ztrue_arr, integ_mass_arr, integ_lambda_true_arr
            ),
            # volume element at each point of z array
            "dvdzdomega_z1z2": derived_cosmology.dV_dzdO(
                self.halo_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            ),
            # hmf at the center of observed redshift bins
            "dndm_z": self.hmfbias.dn_dm(integ_ztrue_arr, integ_mass_arr),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias_z": self.hmfbias.bias(integ_ztrue_arr, integ_mass_arr),
        }

    def _compute_Plob_M_z_in_bin(self, lambda_min, lambda_max, integral_n_steps=31):
        """compute Plob_M_z_bin.
        Compute the probability of the observed richness
        given true mass and redshift P(lambda_obs|mass, z) for a richness bin.

        Parameters
        ----------
        lambda_min : float
            Lower richness edge of the integration bin
        lambda_max : float
            Upper richness edge of the integration bin
        integral_n_steps : int
            Number of points to be used for the interpolation in the integral

        Returns
        -------
        Plob_M_z : numpy.ndarray
            P(lambda_obs|mass, z)
        """
        l_tab = np.geomspace(lambda_min, lambda_max, integral_n_steps)
        # P(lob|ltr,ztr)
        Plob_l_z = simps(
            self.selectionfunction.P_lbdobs_lbd(
                self.cluster_stat_kernel_tables["ztrue"],
                self.cluster_stat_kernel_tables["lambda_true"],
                l_tab,
            ),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF' :
        #       Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.cluster_stat_kernel_tables["ztrue"], self.cluster_stat_kernel_tables["lambda_true"]).T

        # P(lambda_obs|mass, z)
        Plob_M_z = simps(
            self.cluster_stat_kernel_tables["Pltrue_M_z"][:, :, :]
            * Plob_l_z[:, np.newaxis, :],
            x=self.cluster_stat_kernel_tables["lambda_true"],
            axis=-1,
        )
        return Plob_M_z

    def _compute_volume_in_bin(self, z_min, z_max, lambda_min, integral_n_steps):
        """compute volume bin.
        Computes volume in a given richness redshift bin for cluster counts.

        Parameters
        ----------
        z_min : float
            Lower redshift edge of the integration bin
        z_max : float
            Upper redshift edge of the integration bin
        lambda_min : float
            Lower richness edge of the integration bin
        integral_n_steps : int
            Number of points to be used for z_obs integration.

        Returns
        -------
        dV_dzob_bin : numpy.ndarray
            Observed volume element dV/dz_ob in the redshift bin
        """

        # P(zob|ztr)
        z_tab = np.linspace(z_min, z_max, integral_n_steps)
        Pzob_z = simps(
            self.selectionfunction.P_zobs_z(
                z_tab, lambda_min, self.cluster_stat_kernel_tables["ztrue"]
            ),
            x=z_tab,
            axis=0,
        )
        # observed volume element dV/dz_ob
        dV_dzob_bin = (
            self.cluster_stat_kernel_tables["dvdzdomega_z1z2"]
            * Pzob_z
            * (self.area)
            * (np.pi**2.0 / 180.0**2.0)
        )
        # N(lob,zob)
        return dV_dzob_bin

    def _compute_counts_in_bin(self, dV_dzob_bin, nc_lbdobs_z):
        """compute counts bin.
        Compute cluster counts in a single redshift and richness bin
        Performs integral over z_true of the the volume*nc_lbdobs_z

        Parameters
        ----------
        dV_dzob_bin : numpy.ndarray
            volume element of bin
        nc_lbdobs_z : numpy.ndarray
            density of clusters with observed richness and true redshift

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        # computes counts in a richness redshift bin
        return simps(
            nc_lbdobs_z * dV_dzob_bin,
            x=self.cluster_stat_kernel_tables["ztrue"],
            axis=0,
        )

    def compute_binned_quantities(
        self, z_obs_bins, lambda_obs_bins, z_tab_sig=None, l_m_tab_sig=None
    ):
        """Computes binned quantities (counts+aux).

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.
        z_tab_sig : int, None
            Number of points to be used for z_obs integration.
            If None, self.z_tab_sig is used.
        l_m_tab_sig : List, None
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_bins.
            If None, self.l_m_tab_sig is used.

        Returns
        -------
        nc_zbin_lbin : numpy.ndarray
            Number counts in redshift and richness bins
        aux : dict
            Dictionary with intermidate products that can be used for other computations.
            Contains :

                * Plob_M_z (numpy.ndarray) : Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
                * dV_dzob (numpy.ndarray) : Observed volume element (dV/dz_ob) in each redshift and richness bin
                * nc_lbdobs_z (numpy.ndarry) : integral of Plob_M_z*dndm_z on mass.
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        # for z_obs/lambda_obs integration
        if z_tab_sig is None:
            z_tab_sig = self.z_tab_sig
        if l_m_tab_sig is None:
            l_m_tab_sig = self.l_m_tab_sig

        # outputs
        Plob_M_z = np.zeros(
            (
                lambda_obs_bins_size,
                self.cluster_stat_kernel_tables["ztrue"].size,
                self.cluster_stat_kernel_tables["mass"].size,
            )
        )
        dV_dzob = np.zeros((z_obs_bins_size, lambda_obs_bins_size), dtype=list)
        nc_lbdobs_z = np.zeros(lambda_obs_bins_size, dtype=list)
        nc_zbin_lbin = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        for ind_lambda in range(lambda_obs_bins_size):

            Plob_M_z[ind_lambda] = self._compute_Plob_M_z_in_bin(
                lambda_obs_bins[ind_lambda],
                lambda_obs_bins[ind_lambda + 1],
                integral_n_steps=l_m_tab_sig[ind_lambda],
            )
            # N(lob,ztr)
            nc_lbdobs_z[ind_lambda] = simps(
                Plob_M_z[ind_lambda] * self.cluster_stat_kernel_tables["dndm_z"],
                x=self.cluster_stat_kernel_tables["mass"],
                axis=1,
            )

            for ind_z in range(z_obs_bins_size):

                dV_dzob[ind_z, ind_lambda] = self._compute_volume_in_bin(
                    z_obs_bins[ind_z],
                    z_obs_bins[ind_z + 1],
                    lambda_obs_bins[ind_lambda],
                    integral_n_steps=z_tab_sig,
                )
                nc_zbin_lbin[ind_z, ind_lambda] = self._compute_counts_in_bin(
                    dV_dzob[ind_z, ind_lambda], nc_lbdobs_z[ind_lambda]
                )

        aux = {"Plob_M_z": Plob_M_z, "dV_dzob": dV_dzob, "nc_lbdobs_z": nc_lbdobs_z}
        return nc_zbin_lbin, aux

    # -------------------
    # cluster counts cov
    # -------------------

    def compute_binned_bias(self, Plob_M_z, dV_dzob):
        """compute bias.
        Compute halo bias used in counts covariance in bins of redshift and richness

        Parameters
        ----------
        Plob_M_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
        dV_dzob : numpy.ndarray
            Observed volume element (dV/dz_ob) in each redshift and richness bin

        Returns
        -------
        hbias_zbin_lbin : numpy.ndarray
            halo bias in bins of z and lambda
        aux : dict
            Dictionary with intermidate products that can be used for other computations.
            Contains :

                * hb_lbdobs_z (numpy.ndarry) : integral of Plob_M_z*dndm_z*bias_z on mass.
        """

        z_obs_bins_size, lambda_obs_bins_size = dV_dzob.shape

        # outputs
        hb_lbdobs_z = np.zeros(lambda_obs_bins_size, dtype=list)
        hbias_zbin_lbin = np.zeros((z_obs_bins_size, lambda_obs_bins_size))

        for ind_lambda in range(lambda_obs_bins_size):

            # N(lob,ztr) * bias(lob,ztr)
            hb_lbdobs_z[ind_lambda] = simps(
                Plob_M_z[ind_lambda]
                * self.cluster_stat_kernel_tables["dndm_z"]
                * self.cluster_stat_kernel_tables["bias_z"],
                x=self.cluster_stat_kernel_tables["mass"],
                axis=1,
            )

            for ind_z in range(z_obs_bins_size):

                # N(lob,zob) * bias(lob,zob)
                hbias_zbin_lbin[ind_z, ind_lambda] = simps(
                    hb_lbdobs_z[ind_lambda] * dV_dzob[ind_z, ind_lambda],
                    x=self.cluster_stat_kernel_tables["ztrue"],
                    axis=0,
                )

        aux = {"hb_lbdobs_z": hb_lbdobs_z}
        return hbias_zbin_lbin, aux

    def _compute_spatial_cov(self, z_obs_bins, z_tab_sig):
        """Computes only spatial part of the covariance.

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        z_tab_sig : int
            Number of points to be used for z_obs integration.

        Returns
        -------
        spatial_cov : numpy.ndarray
            Spatial part of the covariance
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        z_mid = 0.5 * (z_obs_bins[1:] + z_obs_bins[:-1])

        # for z obs integration
        if z_tab_sig is None:
            z_tab_sig = self.z_tab_sig

        # initialization
        spatial_cov = np.zeros((z_obs_bins_size, z_obs_bins_size))
        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()
        # self.rint = np.zeros((z_obs_bins_size,len(self.cluster_stat_kernel_tables["k"]),L+1))

        # power spectrum at the center of observed redshift bins
        pk = self.halo_statistics.matter_power_spectrum(
            z_mid, self.cluster_stat_kernel_tables["k"]
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
                        self.cluster_stat_kernel_tables["k"] ** 2
                        * np.sqrt(pk[ind_z] * pk[: (ind_z + 1)])
                    )
                    * self.covariance.cov_window(ind_z, z_tab, KL),
                    x=self.cluster_stat_kernel_tables["k"],
                    axis=-1,
                )
            )
            spatial_cov[: (ind_z + 1), ind_z] = spatial_cov[ind_z, : (ind_z + 1)]

        return spatial_cov

    def compute_cov(self, z_obs_bins, nc_zbin_lbin, Plob_M_z, dV_dzob, z_tab_sig=None):
        """Computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        nc_zbin_lbin : numpy.ndarray
            Number counts in redshift and richness bins
        Plob_M_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table.
            To be obtained in aux output of self.compute_binned_quantities.
        dV_dzob : numpy.ndarray
            Observed volume element (dV/dz_ob) in each redshift and richness bin.
            To be obtained in aux output of self.compute_binned_quantities.
        z_tab_sig : int, None
            Number of points to be used for z_obs integration.
            If None, self.z_tab_sig is used.

        Returns
        -------
        cov_nc_zbin_lbin : numpy.ndarray
            Covariance number counts in redshift and richness bins
        """

        # halo bias
        hbias_zbin_lbin, _ = self.compute_binned_bias(Plob_M_z, dV_dzob)

        # spatial component of covariance
        if z_tab_sig is None:
            z_tab_sig = self.z_tab_sig
        spatial_cov = self._compute_spatial_cov(z_obs_bins, z_tab_sig)

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

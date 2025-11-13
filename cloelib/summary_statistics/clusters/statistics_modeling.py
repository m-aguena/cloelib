# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.hmf_bias import HMFBias
from cloelib.observables.clusters.selection_function import SelectionFunction

# import jax

"""

## Notes :

- Cluster statistics modeling

"""


class ClusterStatisticsModeling:
    """Object to compute cluster statistics modeling

    Attributes
    ----------
    kernel_tables : dict
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
        integ_k_arr : np.ndarray
            Values of k to be used in integrations, stored in kernel_tables
        integ_mass_arr : np.ndarray
            Values of mass to be used in integrations, stored in kernel_tables
        integ_lambda_true_arr : np.ndarray
            Values of true richness to be used in integrations, stored in kernel_tables
        integ_ztrue_arr : np.ndarray
            Values of true redshift to be used in integrations, stored in kernel_tables
        area : float
            Area of the survey in deg2.
        """
        # observable objects
        self.hmfbias = hmfbias
        self.halo_statistics = self.hmfbias.halo_statistics
        self.selectionfunction = selectionfunction

        # internal values
        self.area = area

        # integration tables
        self.kernel_tables = {
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
                self.kernel_tables["ztrue"],
                self.kernel_tables["lambda_true"],
                l_tab,
            ),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF' :
        #       Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.kernel_tables["ztrue"], self.kernel_tables["lambda_true"]).T

        # P(lambda_obs|mass, z)
        Plob_M_z = simps(
            self.kernel_tables["Pltrue_M_z"][:, :, :] * Plob_l_z[:, np.newaxis, :],
            x=self.kernel_tables["lambda_true"],
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
                z_tab, lambda_min, self.kernel_tables["ztrue"]
            ),
            x=z_tab,
            axis=0,
        )
        # observed volume element dV/dz_ob
        dV_dzob_bin = (
            self.kernel_tables["dvdzdomega_z1z2"]
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
            x=self.kernel_tables["ztrue"],
            axis=0,
        )

    # -------------------------------------
    # external cluster statistics functions
    # -------------------------------------

    def compute_binned_counts(
        self,
        z_obs_bins,
        lambda_obs_bins,
        z_tab_sig,
        l_m_tab_sig,
        return_intermediate_products=True,
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
        l_m_tab_sig : List, None
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_bins.
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

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        # outputs
        Plob_M_z = np.zeros(
            (
                lambda_obs_bins_size,
                self.kernel_tables["ztrue"].size,
                self.kernel_tables["mass"].size,
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
                Plob_M_z[ind_lambda] * self.kernel_tables["dndm_z"],
                x=self.kernel_tables["mass"],
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

        intermediate_products_zbin_lbin = {
            "Plob_M_z": Plob_M_z,
            "dV_dzob": dV_dzob,
            "nc_lbdobs_z": nc_lbdobs_z,
        }
        if return_intermediate_products:
            return nc_zbin_lbin, intermediate_products_zbin_lbin
        return nc_zbin_lbin

    def compute_binned_bias(self, Plob_M_z, dV_dzob, return_intermediate_products=True):
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
        intermediate_products_zbin_lbin : dict
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
                * self.kernel_tables["dndm_z"]
                * self.kernel_tables["bias_z"],
                x=self.kernel_tables["mass"],
                axis=1,
            )

            for ind_z in range(z_obs_bins_size):

                # N(lob,zob) * bias(lob,zob)
                hbias_zbin_lbin[ind_z, ind_lambda] = simps(
                    hb_lbdobs_z[ind_lambda] * dV_dzob[ind_z, ind_lambda],
                    x=self.kernel_tables["ztrue"],
                    axis=0,
                )

        intermediate_products_zbin_lbin = {"hb_lbdobs_z": hb_lbdobs_z}
        if return_intermediate_products:
            return hbias_zbin_lbin, intermediate_products_zbin_lbin
        return hbias_zbin_lbin

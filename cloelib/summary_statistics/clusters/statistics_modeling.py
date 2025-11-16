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
            * p_ltrue_z_m(np.ndarray) : Values for P(lambda_true|M, z) - shape (z, mass, lambda_true)
            * dvdzdOmega_z(np.ndarray) : Values for volume element at each redshift - shape (z)
            * dndm_z_m(np.ndarray) : Values for the halo mass function dn/dmdz - shape (z, mass)
            * bias_z_m(np.ndarray) : Values for the halo bias halo_bias - shape (z, mass)
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
            # P(lambda_true|M,z), this quantity is also used by cluster clustering
            "p_ltrue_z_m": self.selectionfunction.P_lnlbd(
                integ_ztrue_arr, integ_mass_arr, integ_lambda_true_arr
            ),
            # volume element at each point of z array
            "dvdzdOmega_z": derived_cosmology.dV_dzdO(
                self.halo_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            ),
            # hmf at the center of observed redshift bins
            "dndm_z_m": self.hmfbias.dn_dm(integ_ztrue_arr, integ_mass_arr),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias_z_m": self.hmfbias.bias(integ_ztrue_arr, integ_mass_arr),
        }
        self.kernel_tables["dk"] = self.kernel_tables["k"] ** 2.0 / (2.0 * np.pi**2)

    def _integrate_in_mass_with_hmf(self, quantity):
        """Integrates quantitty in mass with HMF.

        Parameters
        ----------
        quantity : numpy.ndarray
            Kernel to be integrated, must be shape (ztrue, mass).

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        return simps(
            quantity * self.kernel_tables["dndm_z_m"],
            x=self.kernel_tables["mass"],
            axis=1,
        )

    def _integrate_in_ztrue(self, quantity):
        """Integrate quantity in volume.

        Parameters
        ----------
        quantity : numpy.ndarray
            Kernel to be integrated, must be in shape (ztrue).

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        # computes counts in a richness redshift bin
        return simps(quantity, x=self.kernel_tables["ztrue"], axis=0)

    def _integrate_quantity_in_k(self, quantity):
        """Integrates the quantity in k.

        Parameters
        ----------
        quantity : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_quantity : numpy.ndarray
            Quantity integrated in k, dimension same as input
            minus the last one.
        """
        return simps(quantity, x=self.kernel_tables["k"])

    # -------------------------------------
    # external integration functions
    # -------------------------------------

    def integrate_binned_quantity_in_mass_w_hmf(self, binned_quantity):
        """Integrates in mass with HMF each binned quantity.

        Parameters
        ----------
        binned_quantity : numpy.ndarray
            Binned quantity to be integrated in mass, must be dimension
            (nbins, z, mass, ...) with (z, mass) from self.kernel_tables.

        Returns
        -------
        integrated_binned_quantity : numpy.ndarray
            Quantity integrated in mass with the halo
            mass function for each bin. Dimension (nbin, z),
            with (z) from self.kernel_tables.
        """

        bins_size = len(binned_quantity)

        # outputs
        integrated_binned_quantity = np.zeros(
            (*binned_quantity.shape[:2], *binned_quantity.shape[3:])
        )
        for ind in range(bins_size):
            integrated_binned_quantity[ind] = self._integrate_in_mass_with_hmf(
                binned_quantity[ind]
            )
        return integrated_binned_quantity

    def integrate_2d_binned_quantity_in_true_redshift(self, binned_quantity):
        """Integrates in redshift each 2D binned quantity.

        Parameters
        ----------
        binned_quantity : numpy.ndarray
            2D ninned quantity to be integrated in redhisft, must be dimension
            (nbins1, nbins2, z) with (z) from self.kernel_tables.

        Returns
        -------
        integrated_binned_quantity : numpy.ndarray
            Quantity integrated in true redshift for each bin.
            Dimension (nbin1, nbin2).
        """
        bins1_size, bins2_size = binned_quantity.shape[:2]
        out_shape = (*binned_quantity.shape[:2], *binned_quantity.shape[3:])

        # outputs
        integrated_binned_quantity = np.zeros(out_shape)
        for ind1 in range(bins1_size):
            for ind2 in range(bins2_size):
                integrated_binned_quantity[ind1, ind2] = self._integrate_in_ztrue(
                    binned_quantity[ind1, ind2]
                )
        return integrated_binned_quantity

    def integrate_quantity_in_k_space(self, quantity):
        """Integrates the quantity in k space with a k^2/2pi kernel.

        Parameters
        ----------
        quantity : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_quantity : numpy.ndarray
            Quantity integrated in k space, dimension same as input
            minus the last one.
        """
        return self._integrate_quantity_in_k(quantity * self.kernel_tables["dk"])

    # -------------------------------------
    # external cluster statistics functions
    # -------------------------------------

    def compute_binned_volume_element(self, z_obs_bins, lambda_obs_bins, z_tab_sig):
        """Computes volume element in redshift and richness bins.

        Parameters
        ----------
        z_obs_bins : numpy.ndarray
            Redshift bins for the integration.
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.
        z_tab_sig : int, None
            Number of points to be used for z_obs integration.

        Returns
        -------
        dvdz_zbin_lbin_z : numpy.ndarray
            Observed volume element (dV/dz) in each redshift and richness bin
            shape (z_obs, lambda_obs, z) with (z) in kenel_tables.
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        # for z_obs integration
        z_obs_tabs = np.linspace(z_obs_bins[:-1], z_obs_bins[1:], z_tab_sig)

        # outputs
        dvdz_zbin_lbin_z = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, self.kernel_tables["ztrue"].size)
        )
        for ind_z in range(z_obs_bins_size):
            for ind_lambda in range(lambda_obs_bins_size):
                # P(zob|ztr)
                p_zobs_z = simps(
                    self.selectionfunction.P_zobs_z(
                        z_obs_tabs[:, ind_z],
                        lambda_obs_bins[ind_lambda],
                        self.kernel_tables["ztrue"],
                    ),
                    x=z_obs_tabs[:, ind_z],
                    axis=0,
                )
                # observed volume element dV/dz
                dvdz_zbin_lbin_z[ind_z, ind_lambda] = (
                    self.kernel_tables["dvdzdOmega_z"]
                    * p_zobs_z
                    * (self.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )
        return dvdz_zbin_lbin_z

    def compute_binned_lambda_obs_probability(self, lambda_obs_bins, l_m_tab_sig):
        """Computes the probability of observed richness bin P(lobs_bin|M, z)
        with masses and redshifts being the values in self.kernel_tables.

        Parameters
        ----------
        lambda_obs_bins : numpy.ndarray
            Richness bins for the integration.
        l_m_tab_sig : List, None
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_bins.

        Returns
        -------
        p_lbin_z_m : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, z, mass)
        """

        # if external_richness_selection_function == 'CG_ESF' :
        #     p_lbin_z_m  = self.int_Plobltr_Dlob[lambda_bin](self.kernel_tables["ztrue"], self.kernel_tables["lambda_true"]).T

        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        p_lbin_z_m = np.zeros(
            (
                lambda_obs_bins_size,
                self.kernel_tables["ztrue"].size,
                self.kernel_tables["mass"].size,
            )
        )
        for ind_lambda in range(lambda_obs_bins_size):
            # P(lambda_obs_bin|lambda_true, z)
            l_tab = np.geomspace(
                lambda_obs_bins[ind_lambda],
                lambda_obs_bins[ind_lambda + 1],
                l_m_tab_sig[ind_lambda],
            )
            p_lbin_z_ltrue = simps(
                self.selectionfunction.P_lbdobs_lbd(
                    self.kernel_tables["ztrue"],
                    self.kernel_tables["lambda_true"],
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )
            # P(lambda_obs_bin|mass, z)
            p_lbin_z_m[ind_lambda] = simps(
                self.kernel_tables["p_ltrue_z_m"] * p_lbin_z_ltrue[:, np.newaxis, :],
                x=self.kernel_tables["lambda_true"],
                axis=-1,
            )
        return p_lbin_z_m

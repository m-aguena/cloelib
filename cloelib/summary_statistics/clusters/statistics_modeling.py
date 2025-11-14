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
            * p_ltrue_m_z(np.ndarray) : Values for P(lambda_true|mass, z)
            * dvdzdOmega_z(np.ndarray) : Values for volume element at each redshift
            * dndm_m_z(np.ndarray) : Values for the halo mass function dn/dmdz(mass, z)
            * bias_m_z(np.ndarray) : Values for the halo bias halo_bias(mass, z)
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
            # P(ltr|M,z), this quantity is also used by cluster clustering
            "p_ltrue_m_z": self.selectionfunction.P_lnlbd(
                integ_ztrue_arr, integ_mass_arr, integ_lambda_true_arr
            ),
            # volume element at each point of z array
            "dvdzdOmega_z": derived_cosmology.dV_dzdO(
                self.halo_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            ),
            # hmf at the center of observed redshift bins
            "dndm_m_z": self.hmfbias.dn_dm(integ_ztrue_arr, integ_mass_arr),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias_m_z": self.hmfbias.bias(integ_ztrue_arr, integ_mass_arr),
        }
        self.kernel_tables["dk"] = self.kernel_tables["k"] ** 2.0 / (2.0 * np.pi**2)

    def _compute_p_lob_m_z_in_bin(self, lambda_min, lambda_max, integral_n_steps=31):
        """Computes the probability in a observed richness bin given true mass
        and redshift P(lambda_obs_bin|mass, z).

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
        p_lob_m_z_in_bin : numpy.ndarray
            P(lambda_obs_bin|mass, z)
        """
        l_tab = np.geomspace(lambda_min, lambda_max, integral_n_steps)
        # P(lob|ltr,ztr)
        p_lob_l_z = simps(
            self.selectionfunction.P_lbdobs_lbd(
                self.kernel_tables["ztrue"],
                self.kernel_tables["lambda_true"],
                l_tab,
            ),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF' :
        #       p_lob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.kernel_tables["ztrue"], self.kernel_tables["lambda_true"]).T

        # P(lambda_obs|mass, z)
        p_lob_m_z_in_bin = simps(
            self.kernel_tables["p_ltrue_m_z"][:, :, :] * p_lob_l_z[:, np.newaxis, :],
            x=self.kernel_tables["lambda_true"],
            axis=-1,
        )
        return p_lob_m_z_in_bin

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
        dv_dzob_bin : numpy.ndarray
            Observed volume element dV/dz in the redshift bin
        """

        # P(zob|ztr)
        z_tab = np.linspace(z_min, z_max, integral_n_steps)
        p_zob_z = simps(
            self.selectionfunction.P_zobs_z(
                z_tab, lambda_min, self.kernel_tables["ztrue"]
            ),
            x=z_tab,
            axis=0,
        )
        # observed volume element dV/dz
        dv_dzob_bin = (
            self.kernel_tables["dvdzdOmega_z"]
            * p_zob_z
            * (self.area)
            * (np.pi**2.0 / 180.0**2.0)
        )
        return dv_dzob_bin

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
            quantity * self.kernel_tables["dndm_m_z"],
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
        return simps(
            quantity,
            x=self.kernel_tables["ztrue"],
            axis=0,
        )

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
        return simps(
            quantity,
            x=self.kernel_tables["k"],
        )

    # -------------------------------------
    # external integration functions
    # -------------------------------------

    def integrate_binned_quantity_in_mass_w_hmf(self, binned_quantity):
        """Integrates in mass with HMF each binned quantity.

        Parameters
        ----------
        binned_quantity : numpy.ndarray
            Binned quantity to be integrated in mass, must be dimension
            (nbins, mass, z) with (mass, z) from self.kernel_tables.

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
            (bins_size, self.kernel_tables["ztrue"].size)
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

        # outputs
        dvdz_zbin_lbin_z = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, self.kernel_tables["ztrue"].size)
        )
        for ind_lambda in range(lambda_obs_bins_size):
            for ind_z in range(z_obs_bins_size):
                dvdz_zbin_lbin_z[ind_z, ind_lambda] = self._compute_volume_in_bin(
                    z_obs_bins[ind_z],
                    z_obs_bins[ind_z + 1],
                    lambda_obs_bins[ind_lambda],
                    integral_n_steps=z_tab_sig,
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
        p_lbin_m_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, mass, z)
        """
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        p_lbin_m_z = np.zeros(
            (
                lambda_obs_bins_size,
                self.kernel_tables["ztrue"].size,
                self.kernel_tables["mass"].size,
            )
        )
        for ind_lambda in range(lambda_obs_bins_size):
            p_lbin_m_z[ind_lambda] = self._compute_p_lob_m_z_in_bin(
                lambda_obs_bins[ind_lambda],
                lambda_obs_bins[ind_lambda + 1],
                integral_n_steps=l_m_tab_sig[ind_lambda],
            )
        return p_lbin_m_z

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

                * p_lbin_m_z (numpy.ndarray) : Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
                * dvdz_zbin_lbin_z (numpy.ndarray) : Observed volume element (dV/dz) in each redshift and richness bin
                * p_lbin_z (numpy.ndarry) : integral of p_lbin_m_z*dndm_m_z on mass.
        """
        # outputs
        dvdz_zbin_lbin_z = self.compute_binned_volume_element(
            z_obs_bins, lambda_obs_bins, z_tab_sig
        )
        p_lbin_m_z = self.compute_binned_lambda_obs_probability(
            lambda_obs_bins, l_m_tab_sig
        )
        p_lbin_z = self.integrate_binned_quantity_in_mass_w_hmf(p_lbin_m_z)

        nc_zbin_lbin = self.integrate_2d_binned_quantity_in_true_redshift(
            p_lbin_z[np.newaxis, :] * dvdz_zbin_lbin_z
        )
        if not return_intermediate_products:
            return nc_zbin_lbin
        intermediate_products_zbin_lbin = {
            "p_lbin_m_z": p_lbin_m_z,
            "dvdz_zbin_lbin_z": dvdz_zbin_lbin_z,
            "p_lbin_z": p_lbin_z,
        }
        return nc_zbin_lbin, intermediate_products_zbin_lbin

    def compute_binned_bias(
        self, p_lbin_m_z, dvdz_zbin_lbin_z, return_intermediate_products=True
    ):
        """compute bias.
        Compute halo bias used in counts covariance in bins of redshift and richness

        Parameters
        ----------
        p_lbin_m_z : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, z) for masses and redshifts in table
        dvdz_zbin_lbin_z : numpy.ndarray
            Observed volume element (dV/dz) in each redshift and richness bin

        Returns
        -------
        hbias_zbin_lbin : numpy.ndarray
            halo bias in bins of z and lambda
        intermediate_products_zbin_lbin : dict
            Dictionary with intermidate products that can be used for other computations.
            Contains :

                * hb_lbin_z (numpy.ndarry) : integral of p_lbin_m_z*dndm_m_z*bias_m_z on mass.
        """
        # outputs
        hb_lbin_z = self.integrate_binned_quantity_in_mass_w_hmf(
            p_lbin_m_z * self.kernel_tables["bias_m_z"]
        )
        hbias_zbin_lbin = self.integrate_2d_binned_quantity_in_true_redshift(
            hb_lbin_z[np.newaxis, :] * dvdz_zbin_lbin_z
        )
        if not return_intermediate_products:
            return hbias_zbin_lbin
        intermediate_products_zbin_lbin = {"hb_lbin_z": hb_lbin_z}
        return hbias_zbin_lbin, intermediate_products_zbin_lbin

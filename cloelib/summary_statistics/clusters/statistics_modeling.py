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
            * M (np.ndarray) : Values of mass to be used in integrations
            * lambda_true (np.ndarray) : Values of true richness to be used in integrations
            * ztrue (np.ndarray) : Values of true redshift to be used in integrations
            * Pltrue(ztrue,M,lambda_true) (np.ndarray) : Values for P(lambda_true|M, ztrue)
            * dv/dzdOmega(ztrue) (np.ndarray) : Values for volume element at each redshift
            * dn/dM(ztrue,M) (np.ndarray) : Values for the halo mass function dn/dmdz
            * bias(ztrue,M) (np.ndarray) : Values for the halo bias halo_bias
            * dk (np.ndarray) : Kernel k^2/2*pi^2 to be used in k integrations
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
            "M": integ_mass_arr,  # mass array in Msun h^-1
            "lambda_true": integ_lambda_true_arr,  # true richness array
            "ztrue": integ_ztrue_arr,  # true redshift array
            # P(lambda_true|M,z), this quantity is also used by cluster clustering
            "Pltrue(ztrue,M,lambda_true)": self.selectionfunction.P_lnlbd(
                integ_ztrue_arr, integ_mass_arr, integ_lambda_true_arr
            ),
            # volume element at each point of z array
            "dv/dzdOmega(ztrue)": derived_cosmology.dV_dzdO(
                self.halo_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            ),
            # hmf at the center of observed redshift bins
            "dn/dM(ztrue,M)": self.hmfbias.dn_dm(integ_ztrue_arr, integ_mass_arr),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias(ztrue,M)": self.hmfbias.bias(integ_ztrue_arr, integ_mass_arr),
            # kernel for integration in k
            "dk": integ_k_arr**2.0 / (2.0 * np.pi**2),
        }

    def _integrate_in_mass_with_hmf(self, kernel):
        """Integrates quantitty in mass with HMF.

        Parameters
        ----------
        kernel : numpy.ndarray
            Kernel to be integrated, must be shape (ztrue, M).

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        return simps(
            kernel * self.kernel_tables["dn/dM(ztrue,M)"],
            x=self.kernel_tables["M"],
            axis=1,
        )

    def _integrate_in_ztrue(self, kernel):
        """Integrate kernel in volume.

        Parameters
        ----------
        kernel : numpy.ndarray
            Kernel to be integrated, must be in shape (ztrue).

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        # computes counts in a richness redshift bin
        return simps(kernel, x=self.kernel_tables["ztrue"], axis=0)

    def _integrate_kernel_in_k(self, kernel):
        """Integrates the kernel in k.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k, dimension same as input
            minus the last one.
        """
        return simps(kernel, x=self.kernel_tables["k"])

    # -------------------------------------
    # external integration functions
    # -------------------------------------

    def integrate_binned_kernel_in_mass_w_hmf(self, binned_kernel):
        """Integrates in mass with HMF each binned kernel.

        Parameters
        ----------
        binned_kernel : numpy.ndarray
            Binned kernel to be integrated in mass, must be dimension
            (nbins, ztrue, M, ...) with (ztrue, M) from self.kernel_tables.

        Returns
        -------
        integrated_binned_kernel : numpy.ndarray
            Quantity integrated in mass with the halo
            mass function for each bin. Dimension (nbin, ztrue),
            with (ztrue) from self.kernel_tables.
        """

        bins_size = len(binned_kernel)

        # outputs
        integrated_binned_kernel = np.zeros(
            (*binned_kernel.shape[:2], *binned_kernel.shape[3:])
        )
        for ind in range(bins_size):
            integrated_binned_kernel[ind] = self._integrate_in_mass_with_hmf(
                binned_kernel[ind]
            )
        return integrated_binned_kernel

    def integrate_2d_binned_kernel_in_true_redshift(self, binned_kernel):
        """Integrates in redshift each 2D binned kernel.

        Parameters
        ----------
        binned_kernel : numpy.ndarray
            2D binned kernel to be integrated in redhisft, must be dimension
            (nbins1, nbins2, ztrue) with (ztrue) from self.kernel_tables.

        Returns
        -------
        integrated_binned_kernel : numpy.ndarray
            Quantity integrated in true redshift for each bin.
            Dimension (nbin1, nbin2).
        """
        bins1_size, bins2_size = binned_kernel.shape[:2]
        out_shape = (*binned_kernel.shape[:2], *binned_kernel.shape[3:])

        # outputs
        integrated_binned_kernel = np.zeros(out_shape)
        for ind1 in range(bins1_size):
            for ind2 in range(bins2_size):
                integrated_binned_kernel[ind1, ind2] = self._integrate_in_ztrue(
                    binned_kernel[ind1, ind2]
                )
        return integrated_binned_kernel

    def integrate_kernel_in_k_space(self, kernel):
        """Integrates the kernel in k space with a k^2/2pi kernel.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k space, dimension same as input
            minus the last one.
        """
        return self._integrate_kernel_in_k(kernel * self.kernel_tables["dk"])

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
            shape (z_obs, lambda_obs, ztrue) with (ztrue) in kenel_tables.
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
                    self.kernel_tables["dv/dzdOmega(ztrue)"]
                    * p_zobs_z
                    * (self.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )
        return dvdz_zbin_lbin_z

    def compute_binned_lambda_obs_probability(self, lambda_obs_bins, l_m_tab_sig):
        """Computes the probability of observed richness bin P(lobs_bin|M, ztrue)
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
            Probability of observed richness bin P(lobs_bin|M, ztrue)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, ztrue, M)
        """

        # if external_richness_selection_function == 'CG_ESF' :
        #     p_lbin_z_m  = self.int_Plobltr_Dlob[lambda_bin](self.kernel_tables["ztrue"], self.kernel_tables["lambda_true"]).T

        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        p_lbin_z_m = np.zeros(
            (
                lambda_obs_bins_size,
                self.kernel_tables["ztrue"].size,
                self.kernel_tables["M"].size,
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
            # P(lambda_obs_bin|M, z)
            p_lbin_z_m[ind_lambda] = simps(
                self.kernel_tables["Pltrue(ztrue,M,lambda_true)"]
                * p_lbin_z_ltrue[:, np.newaxis, :],
                x=self.kernel_tables["lambda_true"],
                axis=-1,
            )
        return p_lbin_z_m

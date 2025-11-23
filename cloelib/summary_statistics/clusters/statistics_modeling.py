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
            * dv/dz(ztrue) (np.ndarray) : Values for volume element at each redshift
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
            "dv/dz(ztrue)": derived_cosmology.dV_dzdO(
                self.halo_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            )
            * self.area
            * (np.pi**2.0 / 180.0**2.0),
            # hmf at the center of observed redshift bins
            "dn/dM(ztrue,M)": self.hmfbias.dn_dm(integ_ztrue_arr, integ_mass_arr),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias(ztrue,M)": self.hmfbias.bias(integ_ztrue_arr, integ_mass_arr),
            # kernel for integration in k
            "dk": integ_k_arr**2.0 / (2.0 * np.pi**2),
        }

    # ----------------------------
    # cluster statistics functions
    # ----------------------------

    def compute_binned_redshift_obs_probability(
        self, z_obs_bins, lambda_obs_bins, z_tab_sig
    ):
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
        prob_lambda_obs_bins : numpy.ndarray
            Probability of observed richness bin P(lambda_obs_bin|M, ztrue)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, ztrue, M)
        """

        z_obs_bins_size = len(z_obs_bins) - 1
        lambda_obs_bins_size = len(lambda_obs_bins) - 1

        # for z_obs integration
        z_obs_tabs = np.linspace(z_obs_bins[:-1], z_obs_bins[1:], z_tab_sig)

        # reshape for multiplication
        _z_obs_tabs = z_obs_tabs[:, :, np.newaxis]
        _lambda_obs = lambda_obs_bins[np.newaxis, :-1, np.newaxis]
        _ztrue = self.kernel_tables["ztrue"][np.newaxis, np.newaxis, :]

        # outputs
        prob_z_obs = np.zeros(
            (z_obs_bins_size, lambda_obs_bins_size, self.kernel_tables["ztrue"].size)
        )
        for ind_z in range(z_obs_bins_size):
            prob_z_obs[ind_z] = simps(
                self.selectionfunction.P_zobs_z(
                    _z_obs_tabs[:, ind_z], _lambda_obs, _ztrue
                ),
                x=z_obs_tabs[:, ind_z],
                axis=0,
            )
        return prob_z_obs

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
        prob_lambda_obs_bins : numpy.ndarray
            Probability of observed richness bin P(lobs_bin|M, ztrue)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, ztrue, M)
        """

        # if external_richness_selection_function == 'CG_ESF' :
        #     prob_lambda_obs_bins  = self.int_Plobltr_Dlob[lambda_bin](self.kernel_tables["ztrue"], self.kernel_tables["lambda_true"]).T

        lambda_obs_bins_size = len(lambda_obs_bins) - 1
        prob_lambda_obs_bins = np.zeros(
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
            _prob_lambda_obs_given_lambda_true = simps(
                self.selectionfunction.P_lbdobs_lbd(
                    self.kernel_tables["ztrue"],
                    self.kernel_tables["lambda_true"],
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )
            # P(lambda_obs_bin|M, z)
            prob_lambda_obs_bins[ind_lambda] = simps(
                self.kernel_tables["Pltrue(ztrue,M,lambda_true)"]
                * _prob_lambda_obs_given_lambda_true[:, np.newaxis, :],
                x=self.kernel_tables["lambda_true"],
                axis=-1,
            )
        return prob_lambda_obs_bins

    # ---------------------
    # integration functions
    # ---------------------

    def integrate_in_k(self, kernel):
        """Integrates the kernel in k.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k, dimension same as input
            minus the last one.


        Note
        ----
            This exist as a function on its own for the cluster clustering covariange
            computation, somehow it is faster than using integrate_in_k_space,
            to be investigated.
        """
        return simps(kernel, x=self.kernel_tables["k"])

    def integrate_in_k_space(self, kernel):
        """Integrates the kernel in k space with a k^2/2pi kernel.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the last dimension must be of size (k) from self.kernel_tables.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k space, dimension same as input
            minus the last one.
        """
        return self.integrate_in_k(kernel * self.kernel_tables["dk"])

    def integrate_in_mass(self, kernel, prob_lambda_obs_bins):
        """Integrates in mass with HMF each binned kernel.

        Parameters
        ----------
        kernel : numpy.ndarray
            Kernel to be integrated in mass and convoluted with observed richness bins,
            must be dimension (ztrue, M, ...) with (ztrue, M) from self.kernel_tables.
        prob_lambda_obs_bins : numpy.ndarray
            Probability of observed richness bin P(lambda_obs_bin|M, ztrue)
            with masses and redshifts being the values in self.kernel_tables.
            Dimentions: (lobs_bin, ztrue, M)

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in mass with the halo mass function and convoluted
            with observed richness bins. Dimension (lambda_obs, ztrue, ...),
            with (ztrue) from self.kernel_tables.
        """
        # Add lambda_obs_bins dimension to kernel
        _kernel = kernel[np.newaxis, ...]

        # to make prob_lambda_obs_bins, hmf same shape as kernel
        extra_axes = tuple(range(3, 3 + len(_kernel.shape[3:])))

        # reshape prob_lambda_obs_bins
        _prob_lambda_obs_bins = np.expand_dims(prob_lambda_obs_bins, axis=extra_axes)

        # reshape HMF
        _hmf = np.expand_dims(
            self.kernel_tables["dn/dM(ztrue,M)"], axis=(0, *extra_axes)
        )

        # integral of P(lambda_obs_bins|M, z)*b(z)*dn/dM on mass : (lambda_obs_bins, z)
        integrated_kernel = simps(
            _kernel * _hmf * _prob_lambda_obs_bins,
            x=self.kernel_tables["M"],
            axis=2,
        )
        return integrated_kernel

    def integrate_in_true_redshift(self, kernel, prob_z_obs):
        """Integrates in true volume dv/dz(ztrue) a kernel binned in observed richness,
        in each observed redsfhit bin.

        Parameters
        ----------
        kernel : numpy.ndarray
            Kernel binned in observed richness to be integrated in true redshift, and
            convoluted with observed redshift bins. Must be dimension
            (lambda_obs, ztrue, ...) with (ztrue) from self.kernel_tables.
        prob_z_obs : numpy.ndarray
            Probability of observed redshift bin P(z_obs_bin|lambda_obs, ztrue)
            given a observed richness bin and a true redshift.
            Dimentions: (z_obs, lambda_obs, ztrue, ...) with (ztrue) in kernel_tables.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in true redshift for each observed bin.
            Dimension (z_obs, lambda_obs).
        """
        # Add z_obs_bins dimension to kernel
        _kernel = kernel[np.newaxis, ...]

        # to make prob_z_obs, dvdz same shape as kernel
        extra_axes = tuple(range(3, 3 + len(_kernel.shape[3:])))

        # reshape prob_z_obs
        _prob_z_obs = np.expand_dims(prob_z_obs, axis=extra_axes)

        # reshape dvdz
        _dvdz = np.expand_dims(
            self.kernel_tables["dv/dz(ztrue)"], axis=(0, 1, *extra_axes)
        )

        # output : (z_obs, lambda_obs_bins)
        integrated_kernel = simps(
            _kernel * _dvdz * _prob_z_obs,
            x=self.kernel_tables["ztrue"],
            axis=2,
        )
        return integrated_kernel

"""Cluster covariance calculations."""

import numpy as np
from scipy.integrate import simpson
from scipy.special import eval_legendre, spherical_jn

from cloelib.auxiliary.halo_helpers import photoz_rsd_monopole_correction
from .counts import ClusterCounts
from .clustering import ClusterClustering
from .lensing import ClusterWeakLensing


class Covariance:
    """
    Compute the count or clustering-monopole covariance.

    The summary statistic type determines the calculation used by `compute`.
    Predictions and intermediate products must come from the same model and bins.
    """

    def __init__(self, summary_statistic, z_tab_integ=31, ell_max=20):
        """
        Configure covariance calculations for a summary statistic instance.

        Parameters
        ----------
        summary_statistic : ClusterCounts or ClusterClustering
            Supplies the shared integration model, cosmology, k grid and survey
            area. Construct this object, then call `compute` with the
            prediction's intermediate products.
        z_tab_integ : int, optional
            Radial integration points for the count survey window. Default: 31.
        ell_max : int, optional
            Maximum angular multipole of the count survey window. Default: 20.
            This is not the correlation-function multipole.

        See Also
        --------
        compute : Select and evaluate the appropriate covariance.
        """
        if isinstance(summary_statistic, ClusterWeakLensing):
            raise NotImplementedError("Weak-lensing covariance is not implemented.")
        if not isinstance(summary_statistic, (ClusterCounts, ClusterClustering)):
            raise TypeError("Expected ClusterCounts or ClusterClustering.")
        if not isinstance(z_tab_integ, (int, np.integer)) or z_tab_integ < 2:
            raise ValueError("z_tab_integ must be an integer >= 2.")
        if not isinstance(ell_max, (int, np.integer)) or ell_max < 0:
            raise ValueError("ell_max must be a nonnegative integer.")

        self.summary_statistic = summary_statistic
        self.modeling = summary_statistic.cluster_statitstics_modeling
        self.background = self.modeling.halo_model_properties.background
        self.area = self.modeling.area
        self.k = self.modeling.tabulated_integrands["k"]
        self.L = ell_max
        self.z_tab_integ = z_tab_integ
        self.rint = None

    def compute(self, *, intermediates, prediction=None, z_obs_edges=None):
        """
        Select the covariance from the supplied summary statistic type.

        Parameters
        ----------
        intermediates : dict
            Products returned by get_NC or get_xi0 from the same model and bins.
            Counts require window_lambda_obs and window_z_obs. Clustering
            requires pk0_mean_values, radial_shell_window, radial_shell_volume,
            window_z_obs and cluster_counts. Higher multipoles are not supported.
        prediction : numpy.ndarray, optional
            Count prediction of shape (z_obs_bins, richness_bins), required for
            ClusterCounts. Clustering uses intermediates['cluster_counts'].
        z_obs_edges : numpy.ndarray, optional
            Observed-redshift bin edges from get_NC, required for ClusterCounts.

        Returns
        -------
        numpy.ndarray
            Counts: (z_obs, z_obs, richness, richness).
            Clustering: (z_obs, z_obs, richness_pair, richness_pair, radius, radius).
            These are covariance tensors, not flattened two-dimensional matrices.

        Examples
        --------
        Given a configured counts model and its bin edges:

        >>> prediction, products = counts.get_NC(z_edges, richness_edges)
        >>> covariance = Covariance(counts)
        >>> matrix = covariance.compute(
        ...     prediction=prediction, z_obs_edges=z_edges, intermediates=products
        ... )

        For a configured clustering model:

        >>> xi, products = clustering.get_xi0(z_edges, richness_edges, radius_edges)
        >>> matrix = Covariance(clustering).compute(intermediates=products)
        """
        if isinstance(self.summary_statistic, ClusterCounts):
            if prediction is None or z_obs_edges is None:
                raise ValueError("Counts require prediction and z_obs_edges.")
            z_obs_edges = np.asarray(z_obs_edges)
            prediction = np.asarray(prediction)
            if (
                z_obs_edges.ndim != 1
                or z_obs_edges.size < 2
                or not np.all(np.isfinite(z_obs_edges))
                or np.any(np.diff(z_obs_edges) <= 0)
            ):
                raise ValueError("Redshift edges must be finite and strictly increasing.")
            if prediction.ndim != 2 or prediction.shape[0] != z_obs_edges.size - 1:
                raise ValueError("Count shape does not match the redshift bins.")
            # The spatial-window loop fills this workspace in increasing bin order.
            self.rint = np.zeros((z_obs_edges.size - 1, len(self.k), self.L + 1))
            return self._get_NC_covariance(
                z_obs_edges,
                prediction,
                intermediates["window_lambda_obs"],
                intermediates["window_z_obs"],
            )
        if "pk0_mean_values" not in intermediates:
            raise NotImplementedError(
                "Pass get_xi0 products; higher-multipole covariance is not implemented."
            )
        return self._get_xi_covariance(
            intermediates["pk0_mean_values"],
            intermediates["radial_shell_window"],
            intermediates["radial_shell_volume"],
            intermediates["window_z_obs"],
            intermediates["cluster_counts"],
        )

    def _Kl_coeff(self):
        """
        Coefficients of the spherical harmonics expansion of the angular part of the window function

        Parameters
        ----------
        L: int
           Maximum number at which to evaluate the coefficients

        Returns
        -------
        KL: numpy.ndarray
            Coefficients up to L multipole
        """

        ell = np.linspace(0, self.L, self.L + 1, dtype=int)

        theta = np.arccos(1 - (self.area * (np.pi / 180.0) ** 2.0) / (2 * np.pi))

        KL = (
            np.sqrt(np.pi / (2.0 * ell + 1.0))
            * (
                eval_legendre(ell - 1, np.cos(theta))
                - eval_legendre(ell + 1, np.cos(theta))
            )
            / (2.0 * np.pi * (1 - np.cos(theta)))
        )

        KL[0] = 1 / (2.0 * np.sqrt(np.pi))

        return KL

    def _cov_window(self, iz, z_window_edges, KL):
        """
        Computes the window function between redshifts bins

        Parameters
        ----------
        iz: int
            Index of the redshift bins at which to evaluate the window function
        z_window_edges: tuple
             Redshift boundaries (min, max values) of the window.
        KL: numpy.ndarray
            Spherical harmonic expansion coefficients

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i and j are two redshift bin and k are the wavenumbers
        """
        zarr_iz = np.linspace(z_window_edges[0], z_window_edges[1], self.z_tab_integ)

        rvec = self.background.comoving_distance(zarr_iz) * (
            self.background.H0 / 100.0
        )  # Mpc h^{-1}

        Vz = (rvec[-1] ** 3 - rvec[0] ** 3) / 3  # Mpc^3 h^{-3}

        kr = self.k[:, np.newaxis] * rvec

        self.rint[iz] = (
            1
            / Vz
            * simpson(
                rvec**2.0
                * np.array(
                    [spherical_jn(_l, kr, derivative=False) for _l in range(self.L + 1)]
                ),
                x=rvec,
                axis=-1,
            ).T
        )
        return (4 * np.pi) * np.sum(
            self.rint[iz, :, :] * self.rint[: (iz + 1), :, :] * KL[:] ** 2, axis=-1
        )

    def _compute_spatial_cov(self, z_obs_edges):
        """Computes only spatial part of the covariance.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.

        Returns
        -------
        spatial_cov : numpy.ndarray
            Spatial part of the covariance
        """

        z_obs_edges_size = len(z_obs_edges) - 1
        z_mid = 0.5 * (z_obs_edges[1:] + z_obs_edges[:-1])

        # power spectrum at the center of observed redshift bins (z_obs, k)
        pk = self.modeling.halo_model_properties.matter_power_spectrum_cb(
            z_mid, self.modeling.tabulated_integrands["k"]
        )

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        # can neglect richness dependence here
        pk *= photoz_rsd_monopole_correction(
            self.modeling.halo_model_properties.background,
            z_mid,
            self.modeling.tabulated_integrands["k"],
            self.modeling.selection_function.scatter_z_obs(0, z_mid),
        )[0]

        # spherical harmonic expansion coefficients (covariance)
        KL = self._Kl_coeff()

        # compute spatial covariance (z_obs, z_obs)
        spatial_cov = np.zeros((z_obs_edges_size, z_obs_edges_size))
        for ind_z in range(z_obs_edges_size):
            spatial_cov[ind_z, : (ind_z + 1)] = (
                self.modeling.integrate_probe_function_in_dk(
                    np.sqrt(pk[ind_z] * pk[: (ind_z + 1)])
                    * self._cov_window(
                        ind_z,
                        (
                            z_obs_edges[ind_z],
                            z_obs_edges[ind_z + 1],
                        ),
                        KL,
                    ),
                )
            )
            # fill 2nd half of symmetrical matrix
            spatial_cov[: (ind_z + 1), ind_z] = spatial_cov[ind_z, : (ind_z + 1)]
        return spatial_cov

    def _get_NC_covariance(
        self, z_obs_edges, cluster_counts, window_lambda_obs, window_z_obs
    ):
        """Computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins
        window_lambda_obs : numpy.ndarray
            Integral of P(lamda_obs|M, ztrue) in lambda_obs bins.
            Dimensions: (lambda_obs, ztrue, M) with (ztrue, M) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_NC.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
            Dimensions: (z_obs, lambda_obs, ztrue) with (ztrue) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_NC.

        Returns
        -------
        cov_cluster_counts : numpy.ndarray
            Covariance number counts in redshift and richness bins
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
        # cluster integrated bias : (z_obs, lambda_obs)
        halo_bias_mean_values = self.modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.modeling.integrate_probe_function_in_mass(
                self.modeling.tabulated_integrands["bias(ztrue,M)"],
                window_lambda_obs,
            ),
            window_z_obs,
        )

        ####################
        # Compute covraiance
        ####################

        # spatial component of covariance (z_obs, z_obs)
        spatial_cov = self._compute_spatial_cov(z_obs_edges)

        # shot noise (z_obs, z_obs, lambda_obs, lambda_obs)
        _shot_noise = (
            np.diag(cluster_counts.flatten())
            .reshape(*cluster_counts.shape, *cluster_counts.shape)
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance (z_obs, z_obs, lambda_obs, lambda_obs)
        cov_cluster_counts = _shot_noise + (
            halo_bias_mean_values[np.newaxis, :, np.newaxis, :]
            * halo_bias_mean_values[:, np.newaxis, :, np.newaxis]
            * spatial_cov[:, :, np.newaxis, np.newaxis]
        )

        return cov_cluster_counts

    def _get_xi_covariance(
        self,
        pk_mean_values,
        radial_shell_window,
        radial_shell_volume,
        window_z_obs,
        cluster_counts,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        pk_mean_values : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
            Is in the intermediate_integration_products output of get_xi0.
        radial_shell_window : numpy.ndarray
            Cluster count covariance window (z_obs, lambda_obs, k),
            with (k) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi0.
        radial_shell_volume : numpy.ndarray
            Spherical shell volume (z_obs, radius).
            Is in the intermediate_integration_products output of get_xi0.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
            Dimensions: (z_obs, lambda_obs, ztrue) with (ztrue) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi0.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins

        Returns
        -------
        cov_cluster_clustering : numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_edges_size, lambda_obs_edges_size = cluster_counts.shape
        _, radius_edges_size = radial_shell_volume.shape

        ########################################
        # Cluster statistics modeling quantities
        ########################################

        # Compute observed volume in each redshift bin : (z_obs, lambda_obs)
        volume_mean_values = self.modeling.integrate_probe_function_in_redshift(
            np.ones((1, 1)), window_z_obs
        )
        # Compute output shot-noise terms : (z_obs, lambda_obs, lambda_obs)
        vol_over_cluster_counts = (
            volume_mean_values[:, :, np.newaxis]
            / cluster_counts[:, :, np.newaxis]
            * np.identity(lambda_obs_edges_size)[np.newaxis, :, :]
        )

        #############################
        # Compute nuisance parameters
        #############################

        #    alpha(ztrue,lambda_obs), beta(ztrue,lambda_obs),
        #    gamma(ztrue,lambda_obs) are nuisance parameters to be fitted on
        #    (few, ~100) simulations to correct for bias model inaccuracy,
        #    non-poissonian shot-noise and high-order terms ref values are
        #    alpha=0, beta=1, gamma=0 (see Euclid Collaboration : Fumagalli et
        #    al. 2022)
        alpha = np.zeros((z_obs_edges_size, lambda_obs_edges_size))
        beta = np.ones((z_obs_edges_size, lambda_obs_edges_size))
        gamma = np.zeros((z_obs_edges_size, lambda_obs_edges_size))

        # Combine alpha, beta with pk, vol and reshape to be used
        # in cov_g, cov_ng integral
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        beta_pk_mean_values = (
            beta[:, :, np.newaxis, np.newaxis]
            * beta[:, np.newaxis, :, np.newaxis]
            * pk_mean_values
        )
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        avol_bpk_mean_values = (
            # reshape alpha to be (z_obs, lambda_obs, lambda_obs)
            (1 + alpha)[:, :, np.newaxis, np.newaxis]
            * (1 + alpha)[:, np.newaxis, :, np.newaxis]
            * vol_over_cluster_counts[:, :, :, np.newaxis]
            + beta_pk_mean_values
        )

        ####################
        # Compute covariance
        ####################

        # define cluster clustering bin numbers for loops
        lambda_bin_loop = range(lambda_obs_edges_size)
        rad_bin_loop = range(radius_edges_size)

        # cov_g, cov_ng are TWO TERMS OF EQ. 73
        _cov_gaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )
        _cov_nongaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )

        # note : this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_loop:
            for ind_lambda_j in lambda_bin_loop:
                for ind_radius in rad_bin_loop:
                    _cov_nongaussian[
                        :,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_radius,
                        ind_radius,
                    ] = (
                        self.modeling.integrate_probe_function_in_dk(
                            radial_shell_window[:, ind_radius, :]
                            * beta_pk_mean_values[:, ind_lambda_i, ind_lambda_j, :],
                        )
                        * (1 + gamma[:, ind_lambda_i])
                        * vol_over_cluster_counts[:, ind_lambda_i, ind_lambda_i]
                        * (1 + gamma[:, ind_lambda_j])
                        * vol_over_cluster_counts[:, ind_lambda_j, ind_lambda_j]
                        / radial_shell_volume[:, ind_radius]
                    )

                for ind_lambda_k in lambda_bin_loop:
                    for ind_lambda_h in lambda_bin_loop:
                        # somehow using integrate_probe_function_in_k is much faster then
                        # integrate_probe_function_in_dk here, to be investigated

                        # gaussian term
                        _cov_gaussian[
                            :,
                            ind_lambda_i,
                            ind_lambda_j,
                            ind_lambda_k,
                            ind_lambda_h,
                            :,
                            :,
                        ] = self.modeling.integrate_probe_function_in_k(
                            radial_shell_window[:, np.newaxis, :, :]
                            * radial_shell_window[:, :, np.newaxis, :]
                            * avol_bpk_mean_values[
                                :, ind_lambda_i, ind_lambda_k, np.newaxis, np.newaxis, :
                            ]
                            * avol_bpk_mean_values[
                                :, ind_lambda_j, ind_lambda_h, np.newaxis, np.newaxis, :
                            ]
                            * self.modeling.tabulated_integrands["dk"],
                        )

        # Compute the covariance : (z_obs, lambda_obs,  lambda_obs, lambda_obs, lambda_obs, radius, radius)
        # for tranposing lambda_obs_clustering bins
        _invert_index = (0, 1, 2, 4, 3, 5, 6)
        _cov_clustering_4_lambda_obs_edges = (
            (_cov_gaussian + _cov_nongaussian)
            + (_cov_gaussian + _cov_nongaussian).transpose(_invert_index)
        ) / volume_mean_values[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_obs_i, lambda_obs_j, lambda_obs_k, lambda_obs_l) =
        # cov_xi(lambda_obs_j, lambda_obs_i, lambda_obs_l, lambda_obs_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_edges_size)
        # simplify first pair
        _cov_clustering_3_lambda_obs_edges = _cov_clustering_4_lambda_obs_edges[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        _cov_clustering_2_lambda_obs_edges = _cov_clustering_3_lambda_obs_edges[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (z_obs, lambda_obs, lambda_obs, radius, radius),
        # make it (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius),
        # being diagonal in (z_obs, z_obs)
        cov_cluster_clustering = (
            np.identity(z_obs_edges_size)[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * _cov_clustering_2_lambda_obs_edges
        )
        return cov_cluster_clustering

"""

## Notes :

- Clusters clustering class

"""

# General imports
import numpy as np

# cloelib imports
from cloelib.observables.clusters.halo_clustering import HaloClustering
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax


class ClusterClustering:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        clustering: HaloClustering,
        selection_function: SelectionFunction,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        clustering : HaloClusteringCore
            Halo clustering object
        selection_function : SelectionFunction
            Selection function object
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.clustering = clustering
        self.selection_function = selection_function

    def get_xi0(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products=True,
        n_mu=32,
    ):
        """
        Compute the shell-averaged halo correlation-function monopole.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of the observed-redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of the observed-richness bins.
        radius_edges : numpy.ndarray
            Edges of the radial bins.
        return_intermediate_products : bool, optional
            If true, also return the intermediate quantities used in the
            computation. Default is true.
        n_mu : int, optional
            Number of Gauss-Legendre nodes used for the line-of-sight
            integration. Default is 32.

        Returns
        -------
        cluster_multipole : numpy.ndarray
            Shell-averaged monopole in observed-redshift, richness-pair and
            radial bins.
        intermediate_integration_products : dict, optional
            Intermediate quantities used in the computation. Returned only
            when `return_intermediate_products` is true. Contains:

                * pk0_mean_values (numpy.ndarray): Monopole power spectrum
                  averaged over redshift and richness bins.
                * radial_shell_window (numpy.ndarray): Monopole radial
                  shell window.
                * radial_shell_volume (numpy.ndarray): Spherical shell
                  volume.
                * window_z_obs (numpy.ndarray): Integral of
                  P(z_obs|lambda_obs, ztrue) in observed-redshift bins.
                * cluster_counts (numpy.ndarray): Number counts in
                  observed-redshift and richness bins.
        """
        return self._get_xil(
            0,
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            return_intermediate_products,
            n_mu,
        )

    def get_xi2(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products=True,
        n_mu=32,
    ):
        """
        Compute the shell-averaged halo correlation-function quadrupole.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of the observed-redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of the observed-richness bins.
        radius_edges : numpy.ndarray
            Edges of the radial bins.
        return_intermediate_products : bool, optional
            If true, also return the intermediate quantities used in the
            computation. Default is true.
        n_mu : int, optional
            Number of Gauss-Legendre nodes used for the line-of-sight
            integration. Default is 32.

        Returns
        -------
        cluster_multipole : numpy.ndarray
            Shell-averaged quadrupole in observed-redshift, richness-pair and
            radial bins.
        intermediate_integration_products : dict, optional
            Intermediate quantities used in the computation. Returned only
            when `return_intermediate_products` is true. Contains:

                * pk2_mean_values (numpy.ndarray): Quadrupole power spectrum
                  averaged over redshift and richness bins.
                * radial_shell_window (numpy.ndarray): Quadrupole radial
                  shell window.
                * radial_shell_volume (numpy.ndarray): Spherical shell
                  volume.
                * window_z_obs (numpy.ndarray): Integral of
                  P(z_obs|lambda_obs, ztrue) in observed-redshift bins.
                * cluster_counts (numpy.ndarray): Number counts in
                  observed-redshift and richness bins.
        """
        return self._get_xil(
            2,
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            return_intermediate_products,
            n_mu,
        )

    def get_xi4(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products=True,
        n_mu=32,
    ):
        """
        Compute the shell-averaged halo correlation-function hexadecapole.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of the observed-redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of the observed-richness bins.
        radius_edges : numpy.ndarray
            Edges of the radial bins.
        return_intermediate_products : bool, optional
            If true, also return the intermediate quantities used in the
            computation. Default is true.
        n_mu : int, optional
            Number of Gauss-Legendre nodes used for the line-of-sight
            integration. Default is 32.

        Returns
        -------
        cluster_multipole : numpy.ndarray
            Shell-averaged hexadecapole in observed-redshift, richness-pair
            and radial bins.
        intermediate_integration_products : dict, optional
            Intermediate quantities used in the computation. Returned only
            when `return_intermediate_products` is true. Contains:

                * pk4_mean_values (numpy.ndarray): Hexadecapole power spectrum
                  averaged over redshift and richness bins.
                * radial_shell_window (numpy.ndarray): Hexadecapole radial
                  shell window.
                * radial_shell_volume (numpy.ndarray): Spherical shell
                  volume.
                * window_z_obs (numpy.ndarray): Integral of
                  P(z_obs|lambda_obs, ztrue) in observed-redshift bins.
                * cluster_counts (numpy.ndarray): Number counts in
                  observed-redshift and richness bins.
        """
        return self._get_xil(
            4,
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            return_intermediate_products,
            n_mu,
        )

    def _get_xil(
        self,
        ell,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products,
        n_mu,
    ):
        """
        Compute a shell-averaged halo correlation-function multipole.

        Parameters
        ----------
        ell : int
            Even multipole order. Supported values are 0, 2 and 4.
        z_obs_edges : numpy.ndarray
            Edges of the observed-redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of the observed-richness bins.
        radius_edges : numpy.ndarray
            Edges of the radial bins.
        return_intermediate_products : bool
            If true, also return the intermediate quantities used in the
            computation.
        n_mu : int
            Number of Gauss-Legendre nodes used for the line-of-sight
            integration.

        Returns
        -------
        cluster_multipole : numpy.ndarray
            Shell-averaged correlation-function multipole in
            observed-redshift, richness-pair and radial bins.
        intermediate_integration_products : dict, optional
            Intermediate quantities used in the computation. Returned only
            when `return_intermediate_products` is true.
        """
        if ell not in (0, 2, 4):
            raise ValueError(f"Unsupported multipole ell={ell}. Expected 0, 2 or 4.")

        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            self.selection_function, z_obs_edges, lambda_obs_edges
        )
        window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            self.selection_function, lambda_obs_edges
        )
        number_density = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_lambda_obs
            )
        )
        bias_density = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                self.cluster_statitstics_modeling.tabulated_integrands["bias(ztrue,M)"],
                window_lambda_obs,
            )
        )
        cluster_counts = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                number_density, window_z_obs
            )
        )

        z = self.cluster_statitstics_modeling.tabulated_integrands["ztrue"]
        k = self.cluster_statitstics_modeling.tabulated_integrands["k"]
        z_obs_scatter = self.selection_function.scatter_z_obs(
            0.5 * (lambda_obs_edges[1:] + lambda_obs_edges[:-1])[np.newaxis, :],
            z[:, np.newaxis],
        )
        b_eff = (bias_density / number_density).T

        mu, weights = np.polynomial.legendre.leggauss(n_mu)
        pk_mean_values = 0.0
        legendre_coefficients = [0.0] * ell + [1.0]

        for mu_i, weight in zip(mu, weights):
            amplitude = self.clustering.power_spectrum_RSD_amplitude(
                z, k, z_obs_scatter, b_eff, mu_i
            ).transpose(2, 0, 1)
            amplitude = (
                self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                    amplitude * number_density[:, :, np.newaxis],
                    window_z_obs,
                )
                / cluster_counts[:, :, np.newaxis]
            )
            pk_mean_values += (
                0.5
                * (2 * ell + 1)
                * weight
                * np.polynomial.legendre.legval(mu_i, legendre_coefficients)
                * amplitude[:, :, np.newaxis, :]
                * amplitude[:, np.newaxis, :, :]
            )

        radial_shell_window, radial_shell_volume = (
            self.clustering.core.radial_shell_multipole_window_and_volume(
                0.5 * (z_obs_edges[1:] + z_obs_edges[:-1]),
                k,
                radius_edges,
                ell,
            )
        )
        xi = self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
            (-1) ** (ell // 2)
            * radial_shell_window[:, np.newaxis, np.newaxis, :, :]
            * pk_mean_values[:, :, :, np.newaxis, :]
        )

        triangle_indexes = np.triu_indices(len(lambda_obs_edges) - 1)
        cluster_multipole = xi[:, triangle_indexes[0], triangle_indexes[1], :]

        if not return_intermediate_products:
            return cluster_multipole

        intermediate_integration_products = {
            f"pk{ell}_mean_values": pk_mean_values,
            "radial_shell_window": radial_shell_window,
            "radial_shell_volume": radial_shell_volume,
            "cluster_counts": cluster_counts,
            "window_z_obs": window_z_obs,
        }
        return cluster_multipole, intermediate_integration_products

    # ----------------------
    # clustering covariance
    # ----------------------

    def get_xi_covariance(
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
        volume_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                np.ones((1, 1)), window_z_obs
            )
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
                        self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
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
                        ] = self.cluster_statitstics_modeling.integrate_probe_function_in_k(
                            radial_shell_window[:, np.newaxis, :, :]
                            * radial_shell_window[:, :, np.newaxis, :]
                            * avol_bpk_mean_values[
                                :, ind_lambda_i, ind_lambda_k, np.newaxis, np.newaxis, :
                            ]
                            * avol_bpk_mean_values[
                                :, ind_lambda_j, ind_lambda_h, np.newaxis, np.newaxis, :
                            ]
                            * self.cluster_statitstics_modeling.tabulated_integrands[
                                "dk"
                            ],
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

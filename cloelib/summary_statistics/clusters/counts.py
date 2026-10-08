"""

## Notes :

- Cluster counts

"""

# General imports
import numpy as np

# cloelib imports
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax


class ClusterCounts:
    """Object to compute cluster counts"""

    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
    ):
        """
        Initializes the cluster counts

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

    def get_NC(
        self,
        z_obs_edges,
        lambda_obs_edges,
        return_intermediate_products=True,
    ):
        """Computes binned quantities (counts+aux).

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        return_intermediate_products : bool
            If true, also returns `intermediate_integration_products`, a dictionary
            with the intermediate products computed.

        Returns
        -------
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins
        intermediate_integration_products (optional) : dict
            Dictionary with intermidate products that can be used for other computations.
            Returned only when `return_intermediate_products` is true.
            Contains :

                * window_lambda_obs (numpy.ndarray) : Integral of P(lamda_obs|M, ztrue) in lambda_obs bins.
                * window_z_obs (numpy.ndarray) : Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################
        # integral of P(lambda_obs|M, z) on lambda_obs bins : (lambda_obs, M, ztrue)
        window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            lambda_obs_edges
        )
        # integral of P(z_obs|lambda_obs, z) on z_obs bins : (z_obs, lambda_obs, ztrue)
        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            z_obs_edges, lambda_obs_edges
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_lambda_obs
            ),
            window_z_obs,
        )

        if not return_intermediate_products:
            return cluster_counts

        return cluster_counts, {
            "window_lambda_obs": window_lambda_obs,
            "window_z_obs": window_z_obs,
        }

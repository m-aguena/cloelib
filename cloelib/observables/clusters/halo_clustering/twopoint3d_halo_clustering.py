# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn

from cloelib.auxiliary import units
from cloelib.cosmology.cosmology import Background
from cloelib.observables.clusters.auxiliary import (
    photoz_rsd_correction,
)
from cloelib.observables.clusters.halo_clustering.halo_clustering_core import (
    HaloClusteringCore,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class TwoPoint3DHaloClustering:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        background_fid: Background,
        nonu: bool = False,
    ):

        self.core = HaloClusteringCore(matter_statistics, background_fid, nonu)

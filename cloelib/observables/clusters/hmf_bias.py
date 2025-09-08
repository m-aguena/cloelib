# General imports
from typing import Protocol, Union, TypeVar, runtime_checkable

import numpy as np  # type: ignore
import jax.numpy as jnp

"""
## Notes:
 
- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloStatisticsModel(Protocol):
    def f_sigma_nu(self, z: T, M: T) -> T:
        r"""Computes the halo multiplicity function."""
        ...

    def bias(self, z: T, M: T) -> T:
        r"""Computes the halo bias."""
        ...

    def dn_dm(self, z, M):
        r"""Derivative of the number density.

        Computes the derivative of the number density
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dn_dm: numpy.ndarray
            dn_dm[i,j], where i is the redshift axis and j the mass axis.
            Units: h^4 Mpc^{-3} Ms^{-1}.
        """
        ...

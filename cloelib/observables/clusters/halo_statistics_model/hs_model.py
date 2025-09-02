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
        r"""
        Computes the halo multiplicity function.
        """
        ...

    def bias(self, z: T, M: T) -> T:
        r"""
        Computes the halo bias.
        """
        ...

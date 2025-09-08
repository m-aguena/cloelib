# General imports
from typing import Protocol, Union, TypeVar, runtime_checkable

import numpy as np  # type: ignore
import jax.numpy as jnp

from cloelib.cosmology.cosmology import Perturbations

"""
## Notes:
 
- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloStatistics(Protocol):
    """Protocol for computing halo mass function and halo bias."""

    @property
    def background(self):
        r"""Returns the Background class instance"""
        ...

    @property
    def sigma8(self):
        r"""Returns the `sigma_8` value at redshift `z=0`. If `sigma_8`
        is not set as a base parameter, it is computed from the power spectrum.
        """
        ...

    @property
    def nonu(self):
        r"""Includes or not neutrinos on matter density and matter power spectrum."""
        ...

    @nonu.setter
    def nonu(self, value):
        """Set nonu"""
        ...

    @property
    def use_interpolation(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        ...

    @use_interpolation.setter
    def use_interpolation(self, use_interpolation):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        ...

    @property
    def halo_statistics_model(self):
        r"""Returns the Background class instance"""
        ...

    @halo_statistics_model.setter
    def halo_statistics_model(self, value):
        """Set halo_statistics_model"""
        ...

    def _matter_power_spectrum_not_interpolated(self, z, k):
        r"""Computes the non interpolated matter power spectrum.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}

        Returns
        -------
        float or np.ndarray
            Matter power spectrum.
        """
        ...

    def interpolate_matter_power_spectrum(
        self,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""Create internal interpolation of matter power spectrum.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}
        """
        ...

    def window(self, k, R):
        r"""Top-hat window and its derivative.

        Computes the top-hat window function and its derivative.

        Parameters
        ----------
        k: numpy.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}
        R: numpy.ndarray
               Radius where W(kR) is evaluated.
               Units: h^{-1} Mpc

        Returns
        -------
        W: numpy.ndarray
            W[i,j], where i is the wavenumber axis and j the radius axis
        dWdx: numpy.ndarray
            dWdx[i,j], where i is the wavenumber axis and j the radius axis
        """
        ...

    def radius_M(self, M):
        r"""Radius from a mass.

        Converts a mass into a radius.

        Parameters
        ----------
        M: numpy.ndarray
            Mass in h^{-1} Msun

        Returns
        -------
        radius_M: array
            Radius in h^{-1} Mpc
        """
        ...

    def delta_c(self, z):
        r"""Critical overdensity.

        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999).

        Parameters
        ----------
        z: float or np.ndarray
            Redshift at which to evaluate the delta_c

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical overdensity a given redshift.
        """
        ...

    def get_Delta_crit(self, z):
        r"""Critical overdensity factor.

        Converts the input overdensity factor into a critical one.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.

        Returns
        -------
        overdensity: float or np.ndarray
            The overdensity factor which needs
            to be multiplied to the critical
            density in order to define an overdensity.

        Notes
        -----
        The function is returned for :math:`\rm \rho_c` in a density definition
        at a given redshift. The function returns :math:`\rm \Delta` for the
        critical density of the universe, :math:`\rm \Delta \Omega_{m}` for
        the mean matter density of the universe, :math:`\rm \Delta` determined
        by `Bryan & Norman 1998
        <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`_ Equation 6 for
        the virial density.
        """
        ...

    def sigma_z_R(self, z, R):
        r"""Standard deviation of perturbations given a redshift and radius.

        Computes the rms at the radii requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        R: numpy.ndarray
            Radius points in h^{-1} Mpc.

        Returns
        -------
        sigma_z_R: numpy.ndarray
            sigma_z_R[i,j], where i is the redshift axis and j the radius axis.
        """
        ...

    def sigma_z_M(self, z, M):
        r"""Standard deviation of perturbations given a redshift and mass.

        Computes the rms at the masses requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        sigma_z_M: numpy.ndarray
            sigma_z_M[i,j], where i is the redshift axis and j the mass axis.
        """
        ...

    def nu_z_M(self, z, M):
        r"""Peak height.

        Computes the critical overdensity over the rms,
        delta_c/sigma, at a given redshift and mass.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points.

        Returns
        -------
        nu_z_M: numpy.ndarray
            nu_z_M[i,j], where i is the redshift axis and j the mass axis.
        """
        ...

    def dlns_dlnR(self, z, M):
        r"""Derivative of the logarithmic rms.

        Computes the derivative of the log rms
        with respect to the radius
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dlns_dlnR: numpy.ndarray
            dlns_dlnR[i,j], where i is the redshift axis and j the mass axis.
        """
        ...

    def f_sigma_nu(self, z, M):
        r"""Multiplicity function.

        Computes the multiplicity function
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis.

        Note
        ----
        Check the documentation of self.halo_statistics_model.f_sigma_nu for
        details on the model used.
        """
        ...

    def bias(self, z, M):
        r"""Computation of the halo bias.

        Computes the halo bias at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        bias: numpy.ndarray
            bias[i,j], where i is the redshift axis and j the mass axis

        Note
        ----
        Check the documentation of self.halo_statistics_model.f_sigma_nu for
        details on the model used.
        """
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

    def f_sigma_nu(self, z: T, M: T) -> T:
        r"""Computes the halo multiplicity function."""
        ...

    def bias(self, z: T, M: T) -> T:
        r"""Computes the halo bias."""
        ...

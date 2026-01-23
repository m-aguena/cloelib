import numpy as np
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.special import j0, j1

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations

from .auxiliary import convert_distance


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)


class MatterStatistics:
    def __init__(
        self,
        perturbations: Perturbations,
        nonu: bool = False,
        use_interpolation: bool = True,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""Auxiliary class computing quantities used in galaxy cluster models.

        Initialize the class with given perturbations and overdensity definition.
        If requested, cosmological functions (e.g., the matter power spectrum)
        are interpolated.

        Parameters
        ----------
        perturbations : Perturbations
            An object from the `LinearPerturbations` class containing cosmological
            perturbation data (e.g., power spectrum, growth function).
        nonu : bool, optional
            If `True`, massive neutrinos are excluded from the density parameter
            summation.
        use_interpolation : bool, optional
            If true, the class interpolates the matter power spectrum.
            A default interpolation is set when class is instanciated with
            use_interpolation=True. For a more customized interpolation, check
            the interpolate_matter_power_spectrum function.
        """
        self.perturbations = perturbations
        self.nonu = nonu

        # Power spectrum interpolation
        self.Pk_interp = None

        # set P(k) interpolation usage
        self.k = k
        if use_interpolation:
            self.interpolate_matter_power_spectrum(z, self.k)
        self.use_interpolation = use_interpolation

        # interpolate the angular diameter distance
        self.angular_diameter_distance = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(z.min(), z.max() + 1.0e-5, len(z)),
            y=self.background.angular_diameter_distance(
                np.linspace(z.min(), z.max() + 1.0e-5, len(z))
            ),
            ext=2,
        )

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.perturbations.background

    @property
    def nonu(self):
        r"""Includes or not neutrinos on matter density and matter power spectrum."""
        return self.__nonu

    @nonu.setter
    def nonu(self, value):
        """Set nonu"""
        if not isinstance(value, bool):
            raise ValueError(f"value for nonu must be boolean, used {value}")
        self.__nonu = value
        if self.nonu:
            self._Omega_m = self.background.Omega_m_cb
            self._matter_power_spectrum = self.perturbations.matter_power_spectrum_cb
        else:
            self._Omega_m = self.background.Omega_m
            self._matter_power_spectrum = self.perturbations.matter_power_spectrum

    @property
    def use_interpolation(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        return self.__use_interpolation

    @use_interpolation.setter
    def use_interpolation(self, use_interpolation):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        if use_interpolation:
            self.matter_power_spectrum = self.Pk_interp
        else:
            self.matter_power_spectrum = _matter_power_spectrum_not_interpolated
        self.__use_interpolation = use_interpolation

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
        return self._matter_power_spectrum(
            z,
            k,
            hubble_units=True,
            k_hunit=True,
        )

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
        # Power spectrum interpolation
        self.Pk_interp = interpolate.RectBivariateSpline(
            z,
            k,
            self._matter_power_spectrum_not_interpolated(z, k),
        )

    def _generic_mass_density_2h(
        self, R, z, halo_bias, bessel_function, radius_units="Mpc/h"
    ):
        r"""
        Surface or excess surface 2-halo density profile.

        Computes either the cosmological surface or excess surface
        (depending on the input Bessel function) 2-halo density profile.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        bessel_function: function
            Bessel function that goes in the integrand with the power spectrum.
            Used to return the surface density or the excess surface density.
            It should take (ll*theta) as input.

        Returns
        -------
        profile: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        # Calculate base quantities
        D_A = self.angular_diameter_distance(z) * self.background.h

        # Ensure proper array shapes (z, M, R)
        z_outshape = np.asarray(z)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
        D_A_outshape = D_A[:, np.newaxis, np.newaxis]
        rho_m_outshape = (
            self.background.Omega_m(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis, np.newaxis]

        halo_bias_outshape = np.asarray(halo_bias)[:, :, np.newaxis]

        # Two point correlation part

        ## 1. Power spectrum interpolation

        kl_array = self.k

        ## 2. Get radial distance in radians
        _theta = convert_distance(R, radius_units, "radians", D_A[:, np.newaxis])
        theta_outshape = _theta[:, np.newaxis]
        if radius_units.lower() != "mpc/h":
            # in this case, theta_outshape was missing z dimension
            theta_outshape = theta_outshape[np.newaxis, np.newaxis, :, 0]

        ## 3. Integrand function
        def integrand(kl):
            ll = kl * (1.0 + z_outshape) * D_A_outshape
            Pk_vals = self.matter_power_spectrum(z, kl)[:, np.newaxis]
            return bessel_function(ll * theta_outshape) * ll * Pk_vals

        ## 4. Integration
        two_point_corr_outshape = (
            quad_vec(integrand, kl_array.min(), kl_array.max(), epsrel=1e-1)[0]
            * (1.0 + z_outshape)
            * D_A_outshape
        )

        # Final strictly 3D calculation
        profile = (
            1.0e-12 * rho_m_outshape * halo_bias_outshape * two_point_corr_outshape
        ) / (2.0 * np.pi * (1.0 + z_outshape) ** 3.0 * D_A_outshape**2.0)

        return profile

    def surface_mass_density_2h(self, R, z, halo_bias, radius_units="Mpc/h"):
        r"""
        Surface 2-halo density profile.

        Computes the cosmological surface 2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, halo_bias, bessel_function=j0, radius_units=radius_units
        )

    def excess_surface_mass_density_2h(self, R, z, halo_bias, radius_units="Mpc/h"):
        r"""
        Excess surface 2-halo density profile.

        Computes the cosmological excess surface 2-halo
        density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, halo_bias, bessel_function=_bessel_j2, radius_units=radius_units
        )

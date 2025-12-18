import numpy as np
from scipy import interpolate
from scipy.integrate import simpson as simps
from scipy.special import j0, j1
from scipy.stats import skewnorm
from astropy import units as ap_units
from scipy.integrate import quad_vec

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)

class HaloStatistics:
    def __init__(
        self,
        perturbations: Perturbations,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        nonu: bool = False,
        use_interpolation: bool = True,
        z_Pk=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        z_lensing=np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        k=np.geomspace(1e-4, 10, 500),
        zs_max: float = 2.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
    ):
        r"""Auxiliary class computing quantities used in halo mass function and halo bias models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        perturbations : Perturbations
            An object from the `LinearPerturbations` class containing cosmological
            perturbation data (e.g., power spectrum, growth function).
        overdensity_type : str
            Overdensity definition for halo mass calculation. Must be one of:
            - "crit": Relative to critical density of the universe.
            - "mean": Relative to mean matter density.
            - "vir": Virial overdensity from spherical collapse.
        overdensity : int, optional
            Value of the overdensity. Effective for non-virial overdensities.
            Example: If it equals 200, halos are defined as regions with density
            200 times the chosen reference (`crit` or `mean`).
        nonu : bool, optional
            If `True`, massive neutrinos are excluded from the density parameter
            summation.
        use_interpolation : bool, optional
            If true, class uses interpolation for matter power spectrum computation.
            A default interpolation is set when class is instanciated with
            use_interpolation=True. For a more customized interpolation, check
            the interpolate_matter_power_spectrum function.
        """
        self.perturbations = perturbations

        if overdensity_type not in ["crit", "mean", "vir"]:
            raise ValueError("Invalid overdensity definition, %s." % overdensity_type)
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

        self.nonu = nonu

        # Power spectrum interpolation
        self.Pk_interp = None

        # internal value of sigma8
        self.__sigma8 = None

        # set interpolation usage
        self.z = z_lensing
        self.k = k
        if use_interpolation:
            self.interpolate_matter_power_spectrum(z_Pk, self.k)
        self.use_interpolation = use_interpolation

        # interpolate the angular diameter distance
        self.angular_diameter_distance = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(self.z.min(), self.z.max() + 1.0e-5, len(self.z)),
            y=self.background.angular_diameter_distance(
                np.linspace(self.z.min(), self.z.max() + 1.0e-5, len(self.z))
            ),
            ext=2,
        )

        # ???
        self.zs_max = zs_max
        self.mean_nz = mean_nz
        self.sigma_nz = sigma_nz
        self.alpha_nz = alpha_nz

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.z)
        self.nzs = self.n_zs(self.z)

        # to avoid recomputing sigma & dsigmadlnM
        self._tabulated_sigma = {
            "M": None,
            "z": None,
            "values": None,
        }
        self._tabulated_dlnsigmadlnM = {
            "M": None,
            "z": None,
            "values": None,
        }

    def _are_mass_and_z_tabulated(self, z, M, reference_table):
        """Check if mass and redshift are the tabluated values"""
        if any(reference_table[key] is None for key in "Mz"):
            return False
        for name, test_val in (("M", M), ("z", z)):
            if len(reference_table[name]) != len(test_val):
                return False
            elif (reference_table[name] != test_val).any():
                return False
        return True

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.perturbations.background

    @property
    def sigma8(self):
        r"""Returns the `sigma_8` value at redshift `z=0`. If `sigma_8`
        is not set as a base parameter, it is computed from the power spectrum.
        """
        if self.__sigma8 is None:
            self.__sigma8 = self.sigma_z_R([0.0], np.array([8.0]))
        return self.__sigma8

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
        x = R[:, np.newaxis] * k
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3.0
        dWdx = 3.0 * (np.sin(x) * (x**2.0 - 3.0) + 3.0 * x * np.cos(x)) / x**4.0

        return W, dWdx

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
        rho_m_0 = (
            derived_cosmology.rho_crit(self.background, 0.0)
            * self._Omega_m(0.0)
            / self.background.h**2.0
        )
        return (M / rho_m_0 * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

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
        return (
            3.0
            / 20.0
            * (12.0 * np.pi) ** (2.0 / 3.0)
            * (1.0 + 0.012299 * np.log10(self.background.Omega_m(z)))
        )

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
        if self.overdensity_type == "crit":
            Delta = self.overdensity

        elif self.overdensity_type == "mean":
            Delta = self.overdensity * self._Omega_m(z)

        elif self.overdensity_type == "vir":
            x = self._Omega_m(z) - 1.0
            Delta = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

        return Delta

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
        k = self.k  # h/Mpc
        W, _ = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.matter_power_spectrum(z, k).reshape(len(z), 1, len(k))
                    * (W**2.0).reshape(1, len(R), len(k)),
                    x=k,
                    axis=-1,
                )
            )
        )

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

        if not self._are_mass_and_z_tabulated(z, M, self._tabulated_sigma):
            R = self.radius_M(M)  # Mpc/h
            self._tabulated_sigma["M"] = M
            self._tabulated_sigma["z"] = z
            self._tabulated_sigma["values"] = self.sigma_z_R(z, R)

        return self._tabulated_sigma["values"]

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
        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnM(self, z, M):
        r"""Derivative of the logarithmic rms.

        Computes the derivative of the ln rms
        with respect to the ln of mass
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dlns_dlnM: numpy.ndarray
            dlns_dlnM[i,j], where i is the redshift axis and j the mass axis.
        """

        if not self._are_mass_and_z_tabulated(z, M, self._tabulated_dlnsigmadlnM):

            k = self.k  # h/Mpc
            R = self.radius_M(M)  # Mpc/h
            W, dWdx = self.window(k, R)
            dsigma2_dlnR = (
                R
                * np.pi**-2
                * simps(
                    k.reshape(1, 1, len(k)) ** 3
                    * self.matter_power_spectrum(z, k).reshape(len(z), 1, len(k))
                    * W.reshape(1, len(R), len(k))
                    * dWdx.reshape(1, len(R), len(k)),
                    x=k,
                    axis=-1,
                )
            )
            dsigma2_dlnM = dsigma2_dlnR / 3
            sigma = self.sigma_z_M(z, M)

            self._tabulated_dlnsigmadlnM["M"] = M
            self._tabulated_dlnsigmadlnM["z"] = z
            self._tabulated_dlnsigmadlnM["values"] = dsigma2_dlnM / (2 * sigma**2)

        return self._tabulated_dlnsigmadlnM["values"]

    # ------------------------------------
    # Functions specific for mass profiles
    # ------------------------------------

    def convert_distance(
        self, distance, units_in, units_out, angular_diameter_distance=None
    ):
        r"""Convert distances

        Parameters
        ----------
        distance: np.ndarray
            Input projected distances
        units_in: str
            Unit for the input projected distance. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        units_out: str
            Unit for the output projected distance. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        angular_diameter_distance: float, np.ndarray
            Angular diameter distance (units: Mpc/h) to be used for converting
            between angular and physical units. If array, it
            should be in the shape (z.size, 1).

        Returns
        -------
        np.ndarray
            Distance in output units. If z is array and physical to
            angular conversion used, output shape is (z.size, distance.size).
        """
        angular_units_dict = {
            "radians": ap_units.rad,
            "degrees": ap_units.deg,
            "arcmin": ap_units.arcmin,
            "arcsec": ap_units.arcsec,
        }
        _valid_units = ["mpc/h", *angular_units_dict.keys()]
        if units_in.lower() not in _valid_units:
            raise ValueError(f"units_in (={units_in}) must be in {_valid_units}")
        if units_out.lower() not in _valid_units:
            raise ValueError(f"units_out (={units_out}) must be in {_valid_units}")

        if units_in.lower() == units_out.lower():
            return distance

        if units_out.lower() not in angular_units_dict:
            # converting to mpc/h
            theta = (
                (distance * angular_units_dict[units_in]).to(ap_units.rad).value
            )  # distance in radians
            out = theta * angular_diameter_distance
        elif units_in.lower() not in angular_units_dict:
            # converting to angular units
            theta = distance / angular_diameter_distance  # distance in radians
            out = (theta * ap_units.rad).to(angular_units_dict[units_out]).value
        else:
            out = (
                (distance * angular_units_dict[units_in])
                .to(angular_units_dict[units_out])
                .value
            )

        return out

    def sigma_crit(self, z, z_sources):
        r"""
        Critical surface mass density.

        Computes the critical surface mass density at the given
        lens and source redshifts.

        Parameters
        ----------
        z: float
            Lens redshift.
        z_sources: np.ndarray
            Source redshift points.

        Returns
        -------
        sigma_crit : float
            Critical surface mass density (unit: h * Msun / pc^2)
        """
        fact = (units.SPEED_OF_LIGHT / 1.0e3 / units.MPC_TO_KM) ** 2.0 / (
            4.0 * np.pi * units.GRAVITATIONAL_CONSTANT
        )  # Msun/Mpc
        d_a_sources = self.angular_diameter_distance(z_sources)  # Mpc
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_a_lens = self.angular_diameter_distance(z)[:, np.newaxis]  # Mpc
        d_m_lens = (1.0 + z[:, np.newaxis]) * d_a_lens
        d_h = units.SPEED_OF_LIGHT / 1e3 / self.background.H0  # Mpc
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_lens / d_h) ** 2.0)
                - d_m_lens
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_sources / d_h) ** 2.0)
            )
        )
        sig_crit = fact * d_a_sources / (d_a_lens * d_a_lens_source)

        return 1e-12 * sig_crit / self.background.h  # Msun pc^{-2} h

    def n_zs_norM(self, z):
        r"""
        Galaxy number density normalization.

        Computes the galaxy number density normalization given a lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.

        Returns
        -------
        n_zs_norM: float or np.ndarray
            Galaxy number density normalization per redshift
        """
        n_zs_norM = 1.0 / (
            skewnorm.cdf(self.zs_max, self.alpha_nz, self.mean_nz, self.sigma_nz)
            - skewnorm.cdf(z, self.alpha_nz, self.mean_nz, self.sigma_nz)
        )

        return n_zs_norM

    def n_zs(self, z):
        r"""
        Galaxy number density.

        Computes the galaxy number density given a lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.

        Returns
        -------
        n_zs: float or np.ndarray
            Galaxy number density per redshift
        """
        n_zs = np.zeros((z.size, len(self.z)))
        for z_ind, _z in enumerate(z):
            z_s = np.linspace(_z + 1.0e-5, self.zs_max, len(self.z))
            n_zs[z_ind] = skewnorm.pdf(z_s, self.alpha_nz, self.mean_nz, self.sigma_nz)

        return n_zs

    def m_sig_crit_m1(self, z, zbin):
        r"""
        Effective inverse critical surface mass density.

        Computes the effective critical surface mass density at
        the given lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.
        zbin: int
            Index of the lens redshift bin.

        Returns
        -------
        m_sigma_crit_m1: float
            Effective inverse critical surface mass density (units : pc^2 / Msun / h)
        """
        z_s = np.linspace(z + 1.0e-5, self.zs_max, len(self.z), axis=1)
        sig_crit_m1 = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s)

        return self.nzsnorM[zbin] * simps(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def _check_profile_shape(self, z, M, R, profile):
        expected_shape = (
            np.atleast_1d(z).size,
            np.atleast_1d(M).size,
            np.atleast_1d(R).size,
        )
        assert (
            profile.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {profile.shape}"

    def _surface_mass_density_args(self, R, z, M, radius_units="Mpc/h"):
        r"""
        Prepare arguments for _model_surface_mass_density_profile and
        _model_mean_surface_mass_density_profile with correct shapes.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        R_outshape: np.ndarray
            Radius (units: Mpc / h) with shape (1, 1, R.size) if radius_units="Mpc/h"
            else (z.size, 1, R.size)
        RDelta: np.ndarray
            Radius of overdensity (units: Mpc / h) with shape (z.size, M.size, 1)
        densityThreshold: np.ndarray
            Threshold density (units : h * Msun / Mpc**2)  with shape (z.size, 1, 1)
        """
        densityThreshold = np.atleast_1d(
            self.get_Delta_crit(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2.0
        )[:, np.newaxis, np.newaxis]

        RDelta = (
            3.0 * M[np.newaxis, :, np.newaxis] / 4.0 / np.pi / densityThreshold
        ) ** (1.0 / 3.0)

        if radius_units.lower() != "mpc/h":
            D_A = (
                self.background.angular_diameter_distance(z) * self.background.h
            )  # Mpc / h
            R_outshape = self.convert_distance(
                R, radius_units, "Mpc/h", D_A[:, np.newaxis]
            )[:, np.newaxis]
        else:
            R_outshape = R[np.newaxis, np.newaxis, :]

        return R_outshape, RDelta, densityThreshold

    # ----------------------------------
    # Functions with precomputed values
    # ----------------------------------

    def dn_dm_fsigmanu(self, z, M, fsigmanu):
        r"""Derivative of the number density with pre-computed
        halo mass function.

        Computes the derivative of the number density
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.
        fsigmanu: numpy.ndarray
            Multiplicity function.

        Returns
        -------
        dn_dm: numpy.ndarray
            dn_dm[i,j], where i is the redshift axis and j the mass axis.
            Units: h^4 Mpc^{-3} Ms^{-1}.
        """
        rho_mean_0 = self._Omega_m(0) * derived_cosmology.rho_crit(self.background, 0.0)
        rho_mean_0 /= self.background.h**2.0

        return -rho_mean_0 / M**2.0 * fsigmanu * self.dlns_dlnM(z, M)

    def _generic_mass_density_2h(
        self, R, z, M, halo_bias, bessel_function, radius_units="Mpc/h"
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
        M: np.ndarray
            Mass (Msun / h).
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
        D_A = self.background.angular_diameter_distance(z) * self.background.h

        # Ensure proper array shapes (z, M, R)
        z_outshape = np.asarray(z)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
        D_A_outshape = D_A[:, np.newaxis, np.newaxis]
        rho_m_outshape = (
            self.background.Omega_m(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis, np.newaxis]

        # Ensure bias has shape (nz, nM, 1)
        if halo_bias is None:
            raise ValueError(
                "halo_bias must be provided explicitly when computing the 2-halo term."
            )

        halo_bias = np.asarray(halo_bias)
        nz = np.atleast_1d(z).size
        nM = np.atleast_1d(M).size
        if halo_bias.shape != (nz, nM):
            raise ValueError(
                f"halo_bias must have shape (len(z), len(M)) = ({nz}, {nM}), "
                f"got {halo_bias.shape}"
            )
        
        halo_bias_outshape = halo_bias[:, :, np.newaxis]

        # Two point correlation part

        ## 1. Power spectrum interpolation

        kl_array = self.k

        ## 2. Get radial distance in radians
        _theta = self.convert_distance(R, radius_units, "radians", D_A[:, np.newaxis])
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

    def _surface_mass_density_2h(self, R, z, M, halo_bias, radius_units="Mpc/h"):
        r"""
        Surface 2-halo density profile.

        Computes the cosmological surface 2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
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
            R, z, M, halo_bias, bessel_function=j0, radius_units=radius_units
        )

    def _excess_surface_mass_density_2h(
        self, R, z, M, halo_bias, radius_units="Mpc/h"
    ):
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
        M: np.ndarray
            Mass (Msun / h).
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
            R, z, M, halo_bias, bessel_function=_bessel_j2, radius_units=radius_units
        )

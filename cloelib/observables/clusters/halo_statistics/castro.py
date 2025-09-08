import numpy as np
from scipy import interpolate
from scipy.special import gamma

from . import common_functions as cf


class CastroHaloStatistics:
    """Class to compute halo mass function and halo bias with Castro parametrizations."""

    def __init__(
        self,
        perturbations: Perturbations,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        nonu: bool = False,
        use_interpolation: bool = True,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""Initialize the class with given perturbations and overdensity definition.

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
        cf._init_class(
            self,
            perturbations=perturbations,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            nonu=nonu,
            use_interpolation=use_interpolation,
            z=z,
            k=k,
        )

    @property
    def background(self):
        r"""Returns the Background class instance"""
        self.prop.background(self)

    @property
    def sigma8(self):
        r"""Returns the `sigma_8` value at redshift `z=0`. If `sigma_8`
        is not set as a base parameter, it is computed from the power spectrum.
        """
        self.prop.sigma8(self)

    @property
    def nonu(self):
        r"""Includes or not neutrinos on matter density and matter power spectrum."""
        self.prop.nonu(self)

    @nonu.setter
    def nonu(self, value):
        """Set nonu"""
        self.prop.set_nonu(self)

    @property
    def use_interpolation(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        self.prop.use_interpolation(self)

    @use_interpolation.setter
    def use_interpolation(self, use_interpolation):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        self.prop.set_use_interpolation(self)

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
        cf._matter_power_spectrum_not_interpolated(self, z, k)

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
        cf.interpolate_matter_power_spectrum(self, z, k)

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
        cf.window(self, k, R)

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
        cf.radius_M(self, M)

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
        cf.delta_c(self, z)

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
        the virial density.
        """
        cf.get_Delta_crit(self, z)

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
        cf.sigma_z_R(self, z, R)

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
        cf.sigma_z_M(self, z, M)

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
        cf.nu_z_M(self, z, M)

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
        cf.dlns_dlnR(self, z, M)

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
        cf.dn_dm(self, z, M)

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Castro et al. (2023) multiplicity function
        at the requested redshift and mass points.

        Parameters
        ----------
        self: HaloStatistics
            Main halo statistics object.
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis
        """
        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self._Omega_m(z)[:, np.newaxis]
        nu = self.nu_z_M(z, M)

        aR = a1 + a2 * (dlnsigmadlnR + 0.6125) ** 2.0
        a = aR * Ommz**az
        p = p1 + p2 * (dlnsigmadlnR + 0.5)
        qR = q1 + q2 * (dlnsigmadlnR + 0.5)
        q = qR * Ommz**qz
        A = 1.0 / (
            2.0 ** (-0.5 - p + q / 2.0)
            / np.sqrt(np.pi)
            * (2.0**p * gamma(q / 2.0) + gamma(-p + q / 2.0))
        )

        return (
            A
            * np.sqrt(2.0 * a / (np.pi))
            * np.exp(-a * nu**2.0 / 2.0)
            * (1.0 + 1.0 / (a * nu**2.0) ** p)
            * (nu * np.sqrt(a)) ** (q - 1.0)
        ) * nu

    def bias(self, z, M):
        r"""
        Computation of the halo bias.

        Computes the Castro et al. (2024) halo bias
        at the requested redshift and mass points.

        Parameters
        ----------
        self: HaloStatistics
            Main halo statistics object.
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        bias: numpy.ndarray
            bias[i,j], where i is the redshift axis and j the mass axis

        Notes
        -------
        If the mass array has less than 4 entries, this causes problem with the derivative
        """
        if not hasattr(M, "__len__"):
            M = [M]
        M = np.asarray(M)
        lenM_orig = M.size
        if lenM_orig < 4:
            M = np.append(M, M[-1] * np.arange(2, 6))

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self._Omega_m(z)[:, np.newaxis]
        S8 = self.sigma8 * np.sqrt(self.background.Omega_m(0.0) / 0.3)

        nu = self.nu_z_M(z, M)
        nufnu = self.f_sigma_nu(z, M)
        dlnnufnu_dlnnu = np.zeros(nufnu.shape)
        for i in range(len(z)):
            nufnu_int = interpolate.splrep(np.log(nu[i]), np.log(nufnu[i]), s=0)
            dlnnufnu_dlnnu[i] = interpolate.splev(np.log(nu[i]), nufnu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / self.delta_c(z)[:, np.newaxis] * dlnnufnu_dlnnu
        f0 = 1 + a1 * Ommz
        f1 = 1 + b1 * dlnsigmadlnR + b2 * dlnsigmadlnR**2
        f2 = 1 + c1 * S8

        # bias
        bias = A0 * f0 * f1 * f2 * b_pbs

        # original mass array size
        if lenM_orig < len(M):
            bias = bias[:, :lenM_orig]

        return bias

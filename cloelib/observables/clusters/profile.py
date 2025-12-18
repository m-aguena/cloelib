import numpy as np

from cloelib.observables.clusters.halo_statistics import HaloStatistics



class Profile:
    def __init__(
        self,
        halo_statistics: HaloStatistics,
        two_halo: str = "None",
        trunc_fact: float = 3.0,
    ):
        self.halo_statistics = halo_statistics

        if two_halo not in ("None", "sum", "max"):
            raise ValueError("Invalid 'two_halo' definition, %s." % two_halo)
        self.two_halo = two_halo
        self.trunc_fact = trunc_fact

    @property
    def perturbations(self):
        r"""
        Returns the Perturbations class instance
        """
        return self.halo_statistics.perturbations

    @property
    def background(self):
        r"""
        Returns the Background class instance
        """
        return self.perturbations.background

    def surface_mass_density(
        self,
        R,
        z,
        M,
        c,
        halo_bias=None,
        radius_units="Mpc/h",
    ):
        r"""
        Total surface mass density profile.

        Computes the total surface mass density profile at radius R,
        including the contribution from 2-halo term and miscetering.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        c: float
            Concentration.
        halo_bias: np.ndarray (optional)
            Halo bias used for the 2h term, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if self.two_halo != "None" and halo_bias is None:
            raise ValueError("halo_bias must be provided when two_halo != 'None'")

        Sigma = self._surface_mass_density_1h(
            R, z, M, c, radius_units=radius_units
        )

        if self.two_halo != "None":
            Sigma_2h = self.halo_statistics._surface_mass_density_2h(
                R, z, M, halo_bias, radius_units
            )
            if self.two_halo == "sum":
                Sigma += Sigma_2h
            elif self.two_halo == "max":
                Sigma = np.maximum(Sigma, Sigma_2h)

        self.halo_statistics._check_profile_shape(z, M, R, Sigma)
        return Sigma

    def excess_surface_mass_density(
        self, R, z, M, c, halo_bias=None, radius_units="Mpc/h"
    ):
        r"""
        Total excess surface mass density profile.

        Computes the total excess surface mass density profile at radius R,
        including the contribution from 2-halo term and miscetering.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        c: float
            Concentration.
        M: np.ndarray
            Mass (Msun / h).
        halo_bias: np.ndarray (optional)
            Halo bias used for the 2h term, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if self.two_halo != "None" and halo_bias is None:
            raise ValueError("halo_bias must be provided when two_halo != 'None'")

        # centered 1h term
        Sigma_mean = self._model_mean_surface_mass_density_profile(
            *self.halo_statistics._surface_mass_density_args(R, z, M, radius_units=radius_units), c
        )
        Sigma = self._surface_mass_density_1h(
            R, z, M, c, radius_units=radius_units
        )
        DeltaSigma = Sigma_mean - Sigma

        # centered 2h term
        if self.two_halo != "None":
            DeltaSigma_2h = self.halo_statistics._excess_surface_mass_density_2h(
                R, z, M, halo_bias, radius_units
            )
            if self.two_halo == "sum":
                DeltaSigma += DeltaSigma_2h
            elif self.two_halo == "max":
                DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        return DeltaSigma

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        Centered one-halo surface mass density profile.

        Computes the centered one-halo surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            Centered one-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        raise NotImplementedError

    def _surface_mass_density_1h(
        self, R, z, M, c, radius_units="Mpc/h"
    ):
        r"""
        Centered surface mass density profile.

        Computes the centered surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        c: float
            Concentration.
        two_halo: str
            Application of the 2-halo term, options are "None", "sum", "max".
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Centered surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._model_surface_mass_density_profile(
            *self.halo_statistics._surface_mass_density_args(R, z, M, radius_units=radius_units), c
        )

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        Centered one-halo mean surface mass density profile.

        Computes the centered one-halo mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            Centered one-halo mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        raise NotImplementedError

    def _f_term(self, x):
        r"""
        One-halo centered profile F term.

        Computes the one-halo centered profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile F term.
        """
        raise NotImplementedError

    def _g_term(self, x):
        r"""
        One-halo centered profile G term.

        Computes the one-halo centered profile G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile G term.
        """
        raise NotImplementedError


class ProfileNFW(Profile):
    def _f_term(self, x):
        r"""
        NFW profile F term.

        Computes the NFW profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
            One-Halo NFW F term.

        Notes
        -----
        Implementation of second part of Eq. 4 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return (1.0 - np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)) / (x**2.0 - 1.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)) / (x**2.0 - 1.0)

    def _g_term(self, x):
        r"""
        NFW profile G term.

        Computes the NFW profile G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        G_NFW: float
            One-Halo NFW G term.

        Notes
        -----
        Implementation of Eq. 5 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0 + np.log(1.0 / 2.0)
        if x > 1.0:
            return np.log(x / 2.0) + np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        NFW surface mass density profile.

        Computes the NFW surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            NFW surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Rs = RDelta / c
        x = R / Rs

        F = np.vectorize(self._f_term)(x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta * c**3.0 / (3.0 * m_nfw)

        Sigma = 2.0 * rho_s * Rs * F * 1.0e-12

        return Sigma

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        NFW mean surface mass density profile.

        Computes the NFW mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            NFW mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Rs = RDelta / c
        x = R / Rs

        G = np.vectorize(self._g_term)(x)

        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta * c**3.0 / (3.0 * m_nfw)

        return 4.0 * rho_s * Rs * (G / x**2.0) * 1.0e-12


class ProfileBMO(Profile):
    def _f_term(self, x):
        r"""
        BMO profile F term.

        Computes the BMO profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_BMO : float
            One-Halo BMO F term.

        Notes
        -----
        Implementation of Eq. A.5 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0
        if x > 1.0:
            return np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def _g_term(self, x):
        r"""
        BMO profile G term.

        Computes the BMO profile G term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        G_BMO : float
            One-Halo BMO G term.

        Notes
        -----
        Implementation of Eq. A.28 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return (self._f_term(x) - 1.0) / (1.0 - x**2.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - self._f_term(x)) / (x**2.0 - 1.0)

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        BMO surface mass density profile.

        Computes the BMO surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            BMO surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta * c**3.0 / (3.0 * m_bmo)

        const = rho_s_bmo * Rs

        G = np.vectorize(self._g_term)(x)
        F = np.vectorize(self._f_term)(x)

        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0
        term2 = 2.0 * (tau**2.0 + 1.0) * G
        term3 = 8.0 * F
        term4 = (tau**4.0 - 1.0) / (tau**2.0 * (tau**2.0 + x**2.0))
        term5 = (
            np.pi
            * (4.0 * (tau**2.0 + x**2.0) + tau**2.0 + 1.0)
            / (tau**2.0 + x**2.0) ** (3.0 / 2.0)
        )
        term6 = (
            tau**2.0 * (tau**4.0 - 1.0)
            + (tau**2.0 + x**2.0) * (3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0)
        ) / (tau**3.0 * (tau**2.0 + x**2.0) ** (3.0 / 2.0))

        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        Sigma = 1e-12 * const * term1 * (term2 + term3 + term4 - term5 + term6 * L)
        return Sigma

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        BMO mean surface mass density profile.

        Computes the BMO mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            BMO mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta * c**3.0 / (3.0 * m_bmo)

        const = 2.0 * np.pi * rho_s_bmo * Rs**3.0
        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0

        F = np.vectorize(self._f_term)(x)
        term2 = 2.0 * (tau**2.0 + 1.0 + 4.0 * (x**2.0 - 1.0)) * F

        G = np.vectorize(self._g_term)(x)
        term3 = (
            np.pi * (3.0 * tau**2.0 - 1.0) + 2.0 * tau * (tau**2.0 - 3.0) * np.log(tau)
        ) / tau

        term4 = tau**3.0 * np.sqrt(tau**2.0 + x**2.0)
        term5 = -(tau**3.0) * np.pi * (4.0 * (tau**2.0 + x**2.0) - tau**2.0 - 1.0)
        term6 = -(tau**2.0) * (tau**4.0 - 1.0) + +(tau**2.0 + x**2.0) * (
            3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0
        )
        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        M_proj = const * term1 * (term2 + term3 + (term5 + term6 * L) / term4)

        return M_proj / (np.pi * R**2.0) * 1.0e-12

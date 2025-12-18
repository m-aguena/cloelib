import numpy as np

from cloelib.observables.clusters.halo_statistics import HaloStatistics

class NFWMassProfile:

    def __init__(
        self,
        halo_statistics: HaloStatistics,
        two_halo: str = "None",
    ):

        self.halo_statistics = halo_statistics

        if two_halo not in ("None", "sum", "max"):
            raise ValueError("Invalid 'two_halo' definition, %s." % two_halo)
        self.two_halo = two_halo

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

    def _surface_mass_density_1h(self, R, RDelta, Delta, c):
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

    def _mean_surface_mass_density_1h(self, R, RDelta, Delta, c):
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

    def surface_mass_density(
        self, R, z, M, c, halo_bias=None, radius_units="Mpc/h",
    ):
        r"""
        Total surface mass density profile.

        Computes the total surface mass density profile at radius R,
        including the contribution from 2-halo term.

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
        Sigma = self._surface_mass_density_1h(
            *self.halo_statistics._surface_mass_density_args(R, z, M, radius_units=radius_units), c
        )

        if self.two_halo != "None":
            Sigma_2h = self.halo_statistics._surface_mass_density_2h(
                R, z, M, halo_bias, radius_units
            )
            if self.two_halo == "sum":
                Sigma += Sigma_2h
            elif self.two_halo == "max":
                Sigma = np.maximum(Sigma, Sigma_2h)

        expected_shape = (
            np.atleast_1d(z).size,
            np.atleast_1d(M).size,
            np.atleast_1d(R).size,
        )
        assert (
            Sigma.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {Sigma.shape}"

        return Sigma

    def excess_surface_mass_density(
        self, R, z, M, c, halo_bias=None, radius_units="Mpc/h"
    ):
        r"""
        Total excess surface mass density profile.

        Computes the total excess surface mass density profile at radius R,
        including the contribution from 2-halo term.

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
        R_outshape, RDelta, densityThreshold = self.halo_statistics._surface_mass_density_args(R, z, M, radius_units=radius_units)
        Sigma_mean = self._mean_surface_mass_density_1h(
            R_outshape, RDelta, densityThreshold, c
        )
        Sigma = self._surface_mass_density_1h(
            R_outshape, RDelta, densityThreshold, c
        )
        DeltaSigma = Sigma_mean - Sigma

        if self.two_halo != "None":
            DeltaSigma_2h = self.halo_statistics._excess_surface_mass_density_2h(
                R, z, M, halo_bias, radius_units
            )
            if self.two_halo == "sum":
                DeltaSigma += DeltaSigma_2h
            elif self.two_halo == "max":
                DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        return DeltaSigma

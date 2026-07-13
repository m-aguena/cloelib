import numpy as np

from cloelib.auxiliary.cluster_emulators import ClusterEmuNet
from cloelib.observables.clusters.matter_statistics import MatterStatistics

from .emu_net_weights import (
    delta_sigma_max_params as _DEFAULT_DSIGMA_MAX,
    delta_sigma_min_params as _DEFAULT_DSIGMA_MIN,
    delta_sigma_weights as _DEFAULT_DSIGMA_WEIGHTS,
    sigma_max_params as _DEFAULT_SIGMA_MAX,
    sigma_min_params as _DEFAULT_SIGMA_MIN,
    sigma_weights as _DEFAULT_SIGMA_WEIGHTS,
)
from .halo_profile_core import HaloProfileCore
from cloelib.cosmology import derived_cosmology


class MiscBMOHaloProfileEmu:
    r"""
    Miscentered BMO halo profile evaluated via neural-network emulators.

    Two independent six-hidden-layer networks (one for :math:`\Sigma_\mathrm{off}`,
    one for :math:`\Delta\Sigma_\mathrm{off}`) replace the numerical integration
    of the azimuthally averaged miscentered profiles. Each emulator predicts

    .. math::

        \ln\!\left(\frac{\Sigma_\mathrm{off}(R)}{\rho_s}\right)
        \quad\text{or}\quad
        \ln\!\left(\frac{\Delta\Sigma_\mathrm{off}(R)}{\rho_s}\right)

    as a function of the four input features
    :math:`[\log_{10} R,\, \log_{10} R_\mathrm{vir},\, c,\, \sigma_\mathrm{off}]`.

    The characteristic density :math:`\rho_s` is the virial BMO amplitude
    computed analytically from mass, concentration, and redshift (see
    :meth:`_rho_s_bmo`).

    The emulater has been trained setting tau_vir = 3.0 within these boundaries
    # Bounds #log10R [cMpc/h] # log10 Rvir [pMpc/h] # c # sigma_off
    lower_bounds = [-4., np.log10(0.15), 0.5, 0.05]
    upper_bounds = [np.log10(30.), np.log10(2.2), 10., 0.8]

    Parameters
    ----------
    matter_statistics : MatterStatistics
        MatterStatistics object carrying the cosmology.
    overdensity_type : str, optional
        Overdensity type passed to :class:`HaloProfileCore`.  Default ``"vir"``.
    overdensity : int, optional
        Overdensity value passed to :class:`HaloProfileCore`.  Default ``200``.
    trunc_fact : float, optional
        Truncation radius in units of the overdensity radius
        (:math:`R_t = \tau_\mathrm{vir} \cdot R_\Delta`).  Default ``3.0``.
    z : array_like, optional
        Redshift grid for cosmological calculations.
    zs_max : float, optional
        Maximum source redshift for lensing calculations.
    mean_nz : float, optional
        Mean of the source redshift distribution.
    sigma_nz : float, optional
        Width of the source redshift distribution.
    alpha_nz : float, optional
        Shape parameter of the source redshift distribution.

    Notes
    -----
    The emulators were trained following Eq. 8 of
    `Johnston et al. 2007 <https://arxiv.org/pdf/0709.1159.pdf>`_ for the
    miscentering PDF and the BMO truncated NFW profile of
    `Baltz et al. 2009 <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
    """

    def __init__(
        self,
        matter_statistics: MatterStatistics,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        trunc_fact: float = 3.0,
        z: np.ndarray = np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        zs_max: float = 2.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
    ):
        self.core = HaloProfileCore(
            matter_statistics,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            z=z,
            zs_max=zs_max,
            mean_nz=mean_nz,
            sigma_nz=sigma_nz,
            alpha_nz=alpha_nz,
        )

        self.trunc_fact = trunc_fact
        self._emu_sigma = None
        self._emu_delta_sigma = None

    def load_weights(
        self,
        sigma_weights: dict = None,
        delta_sigma_weights: dict = None,
        sigma_min_params: np.ndarray = None,
        sigma_max_params: np.ndarray = None,
        delta_sigma_min_params: np.ndarray = None,
        delta_sigma_max_params: np.ndarray = None,
        hidden_size: int = 512,
    ):
        """
        Instantiate and load the two emulator networks from weight dicts,
        falling back to the weights embedded in EmuNetWeights

        Parameters
        ----------
        sigma_weights : dict, optional
            Weight dictionary for the :math:`\Sigma_\mathrm{off}` emulator,
            with keys ``fc1_w`` … ``fc6_w``, ``fc1_b`` … ``fc6_b``.
            Defaults to the weights embedded in :mod:`EmuNetWeights`.
        delta_sigma_weights : dict, optional
            Weight dictionary for the :math:`\Delta\Sigma_\mathrm{off}`
            emulator.  Defaults to the weights embedded in :mod:`EmuNetWeights`.
        sigma_min_params : array_like, shape (4,)
            Feature minima used to normalise the inputs of the
            :math:`\Sigma_\mathrm{off}` emulator.
            Defaults to the values in :mod:`EmuNetWeights`.
        sigma_max_params : array_like, shape (4,)
            Feature maxima used to normalise the inputs of the
            :math:`\Sigma_\mathrm{off}` emulator.
            Defaults to the values in :mod:`EmuNetWeights`.
        delta_sigma_min_params : array_like, shape (4,)
            Feature minima for the :math:`\Delta\Sigma_\mathrm{off}` emulator.
            Defaults to the values in :mod:`EmuNetWeights`.
        delta_sigma_max_params : array_like, shape (4,)
            Feature maxima for the :math:`\Delta\Sigma_\mathrm{off}` emulator.
            Defaults to the values in :mod:`EmuNetWeights`.
        hidden_size : int, optional
            Hidden-layer width of the emulator networks. Default ``512``.
        """
        self._emu_sigma = ClusterEmuNet(
            input_size=4,
            hidden_size=hidden_size,
            output_size=1,
            x_min=np.asarray(
                sigma_min_params if sigma_min_params is not None else _DEFAULT_SIGMA_MIN
            ),
            x_max=np.asarray(
                sigma_max_params if sigma_max_params is not None else _DEFAULT_SIGMA_MAX
            ),
        )
        self._emu_sigma.load_from_dict(
            sigma_weights if sigma_weights is not None else _DEFAULT_SIGMA_WEIGHTS
        )

        self._emu_delta_sigma = ClusterEmuNet(
            input_size=4,
            hidden_size=hidden_size,
            output_size=1,
            x_min=np.asarray(
                delta_sigma_min_params
                if delta_sigma_min_params is not None
                else _DEFAULT_DSIGMA_MIN
            ),
            x_max=np.asarray(
                delta_sigma_max_params
                if delta_sigma_max_params is not None
                else _DEFAULT_DSIGMA_MAX
            ),
        )
        self._emu_delta_sigma.load_from_dict(
            delta_sigma_weights
            if delta_sigma_weights is not None
            else _DEFAULT_DSIGMA_WEIGHTS
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rho_s_bmo(self, M: np.ndarray, c: float, z: np.ndarray) -> np.ndarray:
        r"""
        BMO characteristic density :math:`\rho_s`.

        Reproduces the analytical :math:`\rho_s` used during emulator training:

        .. math::

            \rho_s = \frac{\Delta_\mathrm{vir}(z)\, c^3}{3\, m_\mathrm{BMO}(c,\tau)}
            \cdot \rho_c(z)

        where :math:`\tau = \tau_\mathrm{vir} \cdot c` and
        :math:`m_\mathrm{BMO}` is the dimensionless BMO mass function
        (Eq. 2 of Johnston et al. 2007 / Eq. A of Baltz et al. 2009).

        Parameters
        ----------
        M : np.ndarray
            Halo mass (Msun / h), shape ``(M.size,)`` or scalar.
        c : float
            Concentration.
        z : np.ndarray
            Redshift, shape ``(z.size,)`` or scalar.

        Returns
        -------
        rho_s : np.ndarray
            Characteristic density (Msun h² / Mpc³),
            shape ``(z.size, M.size)``.
        """

        # Omega_m from cosmology
        Omega_m = self.core.matter_statistics.Omega_m_0
        h = self.core.matter_statistics.background.h

        z = np.atleast_1d(np.asarray(z, dtype=float))  # (Nz,)
        M = np.atleast_1d(np.asarray(M, dtype=float))  # (NM,)

        Ez2 = (
            self.core.matter_statistics.background.hubble_parameter(z) / (h * 100.0)
        ) ** 2.0
        rho_c_z = (
            derived_cosmology.rho_crit(self.core.matter_statistics.background, z)
            / h**2.0
        )  # (Nz,) Msun h^2/pMpc^3

        # Virial overdensity Delta_vir(z) w.r.t. critical density
        # Bryan & Norman (1998) fitting formula
        x = Omega_m * (1.0 + z) ** 3.0 / Ez2 - 1.0  # (Nz,)
        Delta_vir = 18.0 * np.pi**2.0 + 82.0 * x - 39.0 * x**2.0  # (Nz,)

        # Virial radius (Mpc/h), shape (Nz, NM)
        rho_c_z_2d = rho_c_z[:, np.newaxis]  # (Nz, 1)
        Delta_vir_2d = Delta_vir[:, np.newaxis]  # (Nz, 1)
        M_2d = M[np.newaxis, :]  # (1, NM)

        R_vir = (M_2d * 3.0 / (4.0 * np.pi * Delta_vir_2d * rho_c_z_2d)) ** (
            1.0 / 3.0
        )  # (Nz, NM)

        # BMO dimensionless mass function m_bmo(c, tau)
        tau = self.trunc_fact * c  # tau = R_t / R_s = trunc_fact * c

        term1 = (
            c
            * (tau**2.0 + 1.0)
            * (c * (c + 1.0) - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c) - 2.0 * tau**4.0)
        )
        term2 = (
            tau
            * (c + 1.0)
            * (tau**2.0 + c**2.0)
            * (
                2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                + tau
                * (tau**2.0 - 3.0)
                * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
            )
        )
        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (term1 + term2)
        )

        # rho_s = Delta_vir * c³ / (3 * m_bmo) * rho_c(z),  shape (Nz, NM)
        rho_s = rho_c_z_2d * Delta_vir_2d * c**3.0 / (3.0 * m_bmo)

        return rho_s, R_vir  # both (Nz, NM)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _predict(
        self,
        emu: ClusterEmuNet,
        R_mpc: np.ndarray,
        R_vir: np.ndarray,
        c: float,
        sigma_off: float,
        rho_s: np.ndarray,
    ) -> np.ndarray:
        r"""
        Run one emulator over the full (z, M, R) grid.

        Parameters
        ----------
        emu : ClusterEmuNet
            The emulator network to evaluate.
        R_mpc : np.ndarray
            Projected radii (Mpc/h), shape ``(R.size,)``.
        R_vir : np.ndarray
            Virial radii (Mpc/h), shape ``(z.size, M.size)``.
        c : float
            Concentration.
        sigma_off : float
            Miscentering scatter (Mpc/h).
        rho_s : np.ndarray
            Characteristic density (Msun h²/Mpc³), shape ``(z.size, M.size)``.

        Returns
        -------
        profile : np.ndarray
            Predicted profile (h Msun/pc²), shape ``(z.size, M.size, R.size)``.
        """
        # Broadcast each quantity to (Nz, NM, NR)
        inputs = np.column_stack(
            [
                np.log10(R_mpc).flatten(),
                np.broadcast_to(
                    np.log10(R_vir)[:, :, np.newaxis], R_mpc.shape
                ).flatten(),
                np.full(R_mpc.size, c),
                np.full(R_mpc.size, sigma_off),
            ]
        )  # (Nz*NM*NR, 4)

        # run emulator
        emulator_prediction = emu.forward(inputs).flatten()  # (Nz*NM*NR,)

        # Undo log-scaling and multiply by rho_s
        profile = (
            np.exp(emulator_prediction).reshape(R_mpc.shape) * rho_s[:, :, np.newaxis]
        )  # Msun h² / Mpc³ · Mpc

        # Convert to h Msun / pc²: 1 Mpc = 1e6 pc  →  1/Mpc² = 1e-12 /pc²
        return profile * 1.0e-12

    def surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered surface mass density profile :math:`\Sigma_\mathrm{off}`.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (units controlled by ``radius_units``).
        z : np.ndarray
            Lens redshifts, shape ``(Nz,)``.
        M : np.ndarray
            Halo masses (Msun / h), shape ``(NM,)``.
        c : float
            Concentration.
        sigma_off : float
            Scatter of the Rayleigh miscentering PDF (Mpc/h),
            see Eq. 8 of `Johnston et al. 2007
            <https://arxiv.org/pdf/0709.1159.pdf>`_.
        radius_units : str, optional
            Unit for the input radii.  Accepted values are
            ``"Mpc/h"``, ``"radians"``, ``"degrees"``, ``"arcmin"``,
            ``"arcsec"``.  Default ``"Mpc/h"``.

        Returns
        -------
        Sigma_off : np.ndarray
            Miscentered surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        R_mpc, RDelta, _ = self.core.surface_mass_density_args(
            R, z, M, radius_units=radius_units
        )

        rho_s, R_vir = self._rho_s_bmo(M, c, z)

        Sigma_off = self._predict(
            self._emu_sigma,
            R_mpc,
            R_vir,
            c,
            sigma_off,
            rho_s,
        )

        self.core.check_profile_shape(R, z, M, Sigma_off)

        return Sigma_off

    # ------------------------------------------------------------------

    def excess_surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered excess surface mass density profile
        :math:`\Delta\Sigma_\mathrm{off}`.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (units controlled by ``radius_units``).
        z : np.ndarray
            Lens redshifts, shape ``(Nz,)``.
        M : np.ndarray
            Halo masses (Msun / h), shape ``(NM,)``.
        c : float
            Concentration.
        sigma_off : float
            Scatter of the Rayleigh miscentering PDF (Mpc/h),
            see Eq. 8 of `Johnston et al. 2007
            <https://arxiv.org/pdf/0709.1159.pdf>`_.
        radius_units : str, optional
            Unit for the input radii.  Accepted values are
            ``"Mpc/h"``, ``"radians"``, ``"degrees"``, ``"arcmin"``,
            ``"arcsec"``.  Default ``"Mpc/h"``.

        Returns
        -------
        DeltaSigma_off : np.ndarray
            Miscentered excess surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        R_mpc, RDelta, _ = self.core.surface_mass_density_args(
            R, z, M, radius_units=radius_units
        )

        rho_s, R_vir = self._rho_s_bmo(M, c, z)

        DeltaSigma_off = self._predict(
            self._emu_delta_sigma,
            R_mpc,
            R_vir,
            c,
            sigma_off,
            rho_s,
        )

        self.core.check_profile_shape(R, z, M, DeltaSigma_off)

        return DeltaSigma_off

"""Module implementing auxiliary functions"""

import numpy as np
from astropy import units as ap_units
from scipy.special import erf

from cloelib.auxiliary import units


def _evaluate_even_series(x_squared, coefficients):
    r"""Evaluate an even Taylor series using Horner's rule.

    Parameters
    ----------
    x_squared : float or np.ndarray
        Squared expansion variable, :math:`x^2`.
    coefficients : sequence of float
        Taylor coefficients ordered by increasing power of :math:`x^2`.
        For coefficients :math:`(a_0, a_1, \ldots)`, the evaluated series is
        :math:`a_0 + a_1 x^2 + a_2 x^4 + \cdots`.

    Returns
    -------
    float or np.ndarray
        Value of the even Taylor series, with the same shape as
        ``x_squared``.

    Notes
    -----
    Horner's rule is used to reduce the number of arithmetic operations and
    to provide a numerically stable evaluation of the small-argument series
    used by the dispersion-model multipole coefficients.
    """
    result = np.zeros_like(x_squared)
    for coefficient in reversed(coefficients):
        result = result * x_squared + coefficient
    return result


def dispersion_model_monopole_coefficients(k_sigma):
    r"""Dispersion-model coefficients for the power-spectrum monopole.

    Computes the analytic coefficients :math:`A_0`, :math:`B_0`, and
    :math:`C_0` entering the monopole of the Gaussian-damped dispersion
    model.

    Parameters
    ----------
    k_sigma : float or np.ndarray
        Dimensionless product :math:`k\sigma`, where :math:`\sigma` is the
        radial dispersion entering the Gaussian damping term.

    Returns
    -------
    A0, B0, C0 : float or np.ndarray
        Monopole coefficients with the same shape as ``k_sigma``.

    Notes
    -----
    The coefficients are defined through

    .. math::

        P_0(k) = \left[A_0 + B_0\beta + C_0\beta^2\right] b^2 P_m(k),

    with :math:`\beta=f/b`. 

    For :math:`|k\sigma| < 0.5`, even Taylor expansions are used to avoid
    catastrophic cancellation in the exact expressions. For larger
    arguments, the closed-form expressions involving the error function are
    evaluated directly.
    """
    x = np.abs(np.asarray(k_sigma, dtype=float))
    A0 = np.empty_like(x)
    B0 = np.empty_like(x)
    C0 = np.empty_like(x)
    small = x < 0.5

    if np.any(small):
        x_squared = x[small] ** 2
        A0[small] = _evaluate_even_series(
            x_squared,
            (
                1.0,
                -1.0 / 3.0,
                1.0 / 10.0,
                -1.0 / 42.0,
                1.0 / 216.0,
                -1.0 / 1320.0,
                1.0 / 9360.0,
                -1.0 / 75600.0,
                1.0 / 685440.0,
            ),
        )
        B0[small] = _evaluate_even_series(
            x_squared,
            (
                2.0 / 3.0,
                -2.0 / 5.0,
                1.0 / 7.0,
                -1.0 / 27.0,
                1.0 / 132.0,
                -1.0 / 780.0,
                1.0 / 5400.0,
                -1.0 / 42840.0,
                1.0 / 383040.0,
            ),
        )
        C0[small] = _evaluate_even_series(
            x_squared,
            (
                1.0 / 5.0,
                -1.0 / 7.0,
                1.0 / 18.0,
                -1.0 / 66.0,
                1.0 / 312.0,
                -1.0 / 1800.0,
                1.0 / 12240.0,
                -1.0 / 95760.0,
                1.0 / 846720.0,
            ),
        )

    large = ~small
    if np.any(large):
        xl = x[large]
        xl_squared = xl**2
        erf_xl = erf(xl)
        exponential = np.exp(-xl_squared)

        A0[large] = np.sqrt(np.pi) * erf_xl / (2.0 * xl)
        B0[large] = (np.sqrt(np.pi) * erf_xl / 2.0 - xl * exponential) / xl**3
        C0[large] = (
            3.0 * np.sqrt(np.pi) * erf_xl / 8.0
            - xl * (2.0 * xl_squared + 3.0) * exponential / 4.0
        ) / xl**5

    return A0, B0, C0


def dispersion_model_quadrupole_coefficients(k_sigma):
    r"""Dispersion-model coefficients for the power-spectrum quadrupole.

    Computes the analytic coefficients :math:`A_2`, :math:`B_2`, and
    :math:`C_2` entering the quadrupole of the Gaussian-damped dispersion
    model.

    Parameters
    ----------
    k_sigma : float or np.ndarray
        Dimensionless product :math:`k\sigma`, where :math:`\sigma` is the
        radial dispersion entering the Gaussian damping term.

    Returns
    -------
    A2, B2, C2 : float or np.ndarray
        Quadrupole coefficients with the same shape as ``k_sigma``.

    Notes
    -----
    The coefficients are defined through

    .. math::

        P_2(k) = \left[A_2 + B_2\beta + C_2\beta^2\right] b^2 P_m(k),

    with :math:`\beta=f/b`.

    For :math:`|k\sigma| < 0.5`, even Taylor expansions are used to avoid
    catastrophic cancellation in the exact expressions. For larger
    arguments, the closed-form expressions involving the error function are
    evaluated directly.
    """
    x = np.abs(np.asarray(k_sigma, dtype=float))
    A2 = np.empty_like(x)
    B2 = np.empty_like(x)
    C2 = np.empty_like(x)
    small = x < 0.5

    if np.any(small):
        x_squared = x[small] ** 2
        A2[small] = _evaluate_even_series(
            x_squared,
            (
                0.0,
                -2.0 / 3.0,
                2.0 / 7.0,
                -5.0 / 63.0,
                5.0 / 297.0,
                -5.0 / 1716.0,
                1.0 / 2340.0,
                -1.0 / 18360.0,
                1.0 / 162792.0,
            ),
        )
        B2[small] = _evaluate_even_series(
            x_squared,
            (
                4.0 / 3.0,
                -8.0 / 7.0,
                10.0 / 21.0,
                -40.0 / 297.0,
                25.0 / 858.0,
                -1.0 / 195.0,
                7.0 / 9180.0,
                -2.0 / 20349.0,
                1.0 / 89376.0,
            ),
        )
        C2[small] = _evaluate_even_series(
            x_squared,
            (
                4.0 / 7.0,
                -10.0 / 21.0,
                20.0 / 99.0,
                -25.0 / 429.0,
                1.0 / 78.0,
                -7.0 / 3060.0,
                1.0 / 2907.0,
                -1.0 / 22344.0,
                5.0 / 973728.0,
            ),
        )

    large = ~small
    if np.any(large):
        xl = x[large]
        xl_squared = xl**2
        erf_xl = erf(xl)
        exponential = np.exp(-xl_squared)

        A2[large] = (
            5.0
            / (8.0 * xl**3)
            * (
                np.sqrt(np.pi) * (3.0 - 2.0 * xl_squared) * erf_xl
                - 6.0 * xl * exponential
            )
        )
        B2[large] = (
            5.0
            / (8.0 * xl**5)
            * (
                np.sqrt(np.pi) * (9.0 - 2.0 * xl_squared) * erf_xl
                - 2.0 * xl * (4.0 * xl_squared + 9.0) * exponential
            )
        )
        C2[large] = (
            5.0
            / (32.0 * xl**7)
            * (
                3.0 * np.sqrt(np.pi) * (15.0 - 2.0 * xl_squared) * erf_xl
                - 2.0
                * xl
                * (8.0 * (xl_squared + 3.0) * xl_squared + 45.0)
                * exponential
            )
        )

    return A2, B2, C2


def dispersion_model_hexadecapole_coefficients(k_sigma):
    r"""Dispersion-model coefficients for the power-spectrum hexadecapole.

    Computes the analytic coefficients :math:`A_4`, :math:`B_4`, and
    :math:`C_4` entering the hexadecapole of the Gaussian-damped dispersion
    model.

    Parameters
    ----------
    k_sigma : float or np.ndarray
        Dimensionless product :math:`k\sigma`, where :math:`\sigma` is the
        radial dispersion entering the Gaussian damping term.

    Returns
    -------
    A4, B4, C4 : float or np.ndarray
        Hexadecapole coefficients with the same shape as ``k_sigma``.

    Notes
    -----
    The coefficients are defined through

    .. math::

        P_4(k) = \left[A_4 + B_4\beta + C_4\beta^2\right] b^2 P_m(k),

    with :math:`\beta=f/b`. 

    For :math:`|k\sigma| < 0.5`, even Taylor expansions are used to avoid
    catastrophic cancellation in the exact expressions. For larger
    arguments, the closed-form expressions involving the error function are
    evaluated directly.
    """
    x = np.abs(np.asarray(k_sigma, dtype=float))
    A4 = np.empty_like(x)
    B4 = np.empty_like(x)
    C4 = np.empty_like(x)
    small = x < 0.5

    if np.any(small):
        x_squared = x[small] ** 2
        A4[small] = _evaluate_even_series(
            x_squared,
            (
                0.0,
                0.0,
                4.0 / 35.0,
                -4.0 / 77.0,
                2.0 / 143.0,
                -2.0 / 715.0,
                1.0 / 2210.0,
                -1.0 / 16150.0,
                1.0 / 135660.0,
            ),
        )
        B4[small] = _evaluate_even_series(
            x_squared,
            (
                0.0,
                -16.0 / 35.0,
                24.0 / 77.0,
                -16.0 / 143.0,
                4.0 / 143.0,
                -6.0 / 1105.0,
                7.0 / 8075.0,
                -4.0 / 33915.0,
                3.0 / 214130.0,
            ),
        )
        C4[small] = _evaluate_even_series(
            x_squared,
            (
                8.0 / 35.0,
                -24.0 / 77.0,
                24.0 / 143.0,
                -8.0 / 143.0,
                3.0 / 221.0,
                -21.0 / 8075.0,
                2.0 / 4845.0,
                -6.0 / 107065.0,
                3.0 / 450800.0,
            ),
        )

    large = ~small
    if np.any(large):
        xl = x[large]
        xl_squared = xl**2
        xl_fourth = xl**4
        erf_xl = erf(xl)
        exponential = np.exp(-xl_squared)

        A4[large] = (
            9.0
            / (64.0 * xl**5)
            * (
                3.0
                * np.sqrt(np.pi)
                * (4.0 * xl_fourth - 20.0 * xl_squared + 35.0)
                * erf_xl
                - 10.0 * xl * (2.0 * xl_squared + 21.0) * exponential
            )
        )
        B4[large] = (
            9.0
            / (64.0 * xl**7)
            * (
                3.0
                * np.sqrt(np.pi)
                * (4.0 * xl_fourth - 60.0 * xl_squared + 175.0)
                * erf_xl
                - 2.0
                * xl
                * (32.0 * xl_fourth + 170.0 * xl_squared + 525.0)
                * exponential
            )
        )
        C4[large] = (
            9.0
            / (256.0 * xl**9)
            * (
                3.0
                * np.sqrt(np.pi)
                * (12.0 * xl_fourth - 300.0 * xl_squared + 1225.0)
                * erf_xl
                - 2.0
                * xl
                * (64.0 * xl**6 + 416.0 * xl_fourth + 1550.0 * xl_squared + 3675.0)
                * exponential
            )
        )

    return A4, B4, C4


def _photoz_rsd_parameters(background, z, k, z_obs_scatter):
    r"""Compute the growth rate and Gaussian damping argument.

    Parameters
    ----------
    background : Background
        Background cosmology. The object must provide ``Omega_cb(z)``,
        ``hubble_parameter(z)``, and ``H0``.
    z : float or np.ndarray
        Redshift.
    k : float or np.ndarray
        Wavenumber in :math:`h\,\mathrm{Mpc}^{-1}`.
    z_obs_scatter : float or np.ndarray
        Observed redshift scatter, :math:`\sigma_z`. If an array is
        provided, its first dimension must correspond to redshift.

    Returns
    -------
    f_gr : np.ndarray
        Linear growth-rate approximation,
        :math:`f(z)=\Omega_{\mathrm{cb}}(z)^{0.55}`, broadcast over
        wavenumber and any additional scatter dimensions.
    k_sigma : np.ndarray
        Dimensionless damping argument :math:`k\sigma_r`, where

        .. math::

            \sigma_r(z) =
            \frac{c\,\sigma_z}{H(z)}\frac{H_0}{100}.

    Notes
    -----
    The returned arrays are shaped for direct use in the analytic
    dispersion-model multipole coefficients. Massive neutrinos are excluded
    from :math:`\Omega_{\mathrm{cb}}`.
    """
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    z_obs_scatter_arr = np.asarray(z_obs_scatter)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1.0e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100.0)
    )

    if z_obs_scatter_arr.ndim > 1:
        extra_axes = tuple(range(2, z_obs_scatter_arr.ndim + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)

    if z_obs_scatter_arr.ndim > 0:
        z_obs_scatter_arr = z_obs_scatter_arr[:, np.newaxis, ...]

    return f_gr, ks_z * z_obs_scatter_arr


def convert_to_Delta_crit(overdensity_type, overdensity=200, background=None, z=0.0):
    r"""Critical overdensity factor.

    Converts an input halo overdensity definition to an overdensity relative
    to the critical density. The contribution from massive neutrinos is not
    included in the matter density parameter.

    Parameters
    ----------
    overdensity_type : str
        Overdensity definition for the halo mass. Must be one of:
        - ``"crit"``: relative to the critical density of the universe.
        - ``"mean"``: relative to the mean matter density.
        - ``"vir"``: virial overdensity from spherical collapse.
    overdensity : int or float, optional
        Overdensity value for non-virial definitions. For example, ``200``
        defines a density equal to 200 times the selected reference density.
    background : Background, optional
        Background cosmology. Required for ``"mean"`` and ``"vir"``.
    z : float or np.ndarray, optional
        Redshift.

    Returns
    -------
    float or np.ndarray
        Overdensity factor multiplying the critical density.

    Notes
    -----
    For ``"crit"``, the function returns :math:`\Delta`. For ``"mean"``, it
    returns :math:`\Delta\Omega_{\mathrm{cb}}(z)`. For ``"vir"``, it returns
    the Bryan & Norman (1998) spherical-collapse overdensity,

    .. math::

        \Delta_{\mathrm{vir}} =
        18\pi^2 + 82x - 39x^2,

    where :math:`x=\Omega_{\mathrm{cb}}(z)-1`.
    """
    if overdensity_type not in ["crit", "mean", "vir"]:
        raise ValueError("Invalid overdensity definition, %s." % overdensity_type)

    if overdensity_type == "crit":
        return overdensity

    elif overdensity_type == "mean":
        return overdensity * background.Omega_cb(z)

    elif overdensity_type == "vir":
        x = background.Omega_cb(z) - 1.0
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def convert_distance(distance, units_in, units_out, angular_diameter_distance=None):
    r"""Convert projected distances between physical and angular units.

    Parameters
    ----------
    distance : float or np.ndarray
        Input projected distance.
    units_in : str
        Unit of the input distance. Accepted values are ``"Mpc/h"``,
        ``"radians"``, ``"degrees"``, ``"arcmin"``, and ``"arcsec"``.
    units_out : str
        Unit of the output distance. Accepted values are ``"Mpc/h"``,
        ``"radians"``, ``"degrees"``, ``"arcmin"``, and ``"arcsec"``.
    angular_diameter_distance : float or np.ndarray, optional
        Angular-diameter distance in :math:`\mathrm{Mpc}/h`, required when
        converting between physical and angular units. If an array is
        supplied, it should have shape ``(z.size, 1)``.

    Returns
    -------
    float or np.ndarray
        Distance expressed in ``units_out``. When converting between physical
        and angular units with an array of angular-diameter distances, the
        output is broadcast over redshift.

    Raises
    ------
    ValueError
        If ``units_in`` or ``units_out`` is not one of the accepted units.

    Notes
    -----
    Angular-to-physical conversion uses the small-angle relation
    :math:`r_\perp=D_A\theta`.
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


def photoz_rsd_monopole_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Photo-z and redshift-space distortion correction for the monopole.

    Computes the three bias-order contributions to the Gaussian-damped
    dispersion-model monopole.

    Parameters
    ----------
    background : Background
        Background cosmology.
    z : float or np.ndarray
        Redshift.
    k : float or np.ndarray
        Wavenumber in :math:`h\,\mathrm{Mpc}^{-1}`.
    z_obs_scatter : float or np.ndarray
        Observed redshift scatter, :math:`\sigma_z`. If an array is
        provided, its first dimension must correspond to redshift.

    Returns
    -------
    corr0, corr1, corr2 : np.ndarray
        Monopole correction terms multiplying :math:`b^2`, :math:`b`, and
        the bias-independent contribution, respectively. Their values are
        :math:`A_0`, :math:`fB_0`, and :math:`f^2C_0`.

    Notes
    -----
    The resulting monopole can be written as

    .. math::

        P_0(k) =
        \left[b^2\,\mathrm{corr0}
        + b\,\mathrm{corr1}
        + \mathrm{corr2}\right] P_m(k).

    """
    f_gr, k_sigma = _photoz_rsd_parameters(background, z, k, z_obs_scatter)
    A0, B0, C0 = dispersion_model_monopole_coefficients(k_sigma)

    corr0 = A0
    corr1 = f_gr * B0
    corr2 = f_gr**2 * C0

    return corr0, corr1, corr2


def photoz_rsd_quadrupole_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Photo-z and redshift-space distortion correction for the quadrupole.

    Computes the three bias-order contributions to the Gaussian-damped
    dispersion-model quadrupole.

    Parameters
    ----------
    background : Background
        Background cosmology.
    z : float or np.ndarray
        Redshift.
    k : float or np.ndarray
        Wavenumber in :math:`h\,\mathrm{Mpc}^{-1}`.
    z_obs_scatter : float or np.ndarray
        Observed redshift scatter, :math:`\sigma_z`. If an array is
        provided, its first dimension must correspond to redshift.

    Returns
    -------
    corr0, corr1, corr2 : np.ndarray
        Quadrupole correction terms multiplying :math:`b^2`, :math:`b`, and
        the bias-independent contribution, respectively. Their values are
        :math:`A_2`, :math:`fB_2`, and :math:`f^2C_2`.

    Notes
    -----
    The resulting quadrupole can be written as

    .. math::

        P_2(k) =
        \left[b^2\,\mathrm{corr0}
        + b\,\mathrm{corr1}
        + \mathrm{corr2}\right] P_m(k).

    """
    f_gr, k_sigma = _photoz_rsd_parameters(background, z, k, z_obs_scatter)
    A2, B2, C2 = dispersion_model_quadrupole_coefficients(k_sigma)

    corr0 = A2
    corr1 = f_gr * B2
    corr2 = f_gr**2 * C2

    return corr0, corr1, corr2


def photoz_rsd_hexadecapole_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Photo-z and redshift-space distortion correction for the hexadecapole.

    Computes the three bias-order contributions to the Gaussian-damped
    dispersion-model hexadecapole.

    Parameters
    ----------
    background : Background
        Background cosmology.
    z : float or np.ndarray
        Redshift.
    k : float or np.ndarray
        Wavenumber in :math:`h\,\mathrm{Mpc}^{-1}`.
    z_obs_scatter : float or np.ndarray
        Observed redshift scatter, :math:`\sigma_z`. If an array is
        provided, its first dimension must correspond to redshift.

    Returns
    -------
    corr0, corr1, corr2 : np.ndarray
        Hexadecapole correction terms multiplying :math:`b^2`, :math:`b`,
        and the bias-independent contribution, respectively. Their values are
        :math:`A_4`, :math:`fB_4`, and :math:`f^2C_4`.

    Notes
    -----
    The resulting hexadecapole can be written as

    .. math::

        P_4(k) =
        \left[b^2\,\mathrm{corr0}
        + b\,\mathrm{corr1}
        + \mathrm{corr2}\right] P_m(k).
    """
    f_gr, k_sigma = _photoz_rsd_parameters(background, z, k, z_obs_scatter)
    A4, B4, C4 = dispersion_model_hexadecapole_coefficients(k_sigma)

    corr0 = A4
    corr1 = f_gr * B4
    corr2 = f_gr**2 * C4

    return corr0, corr1, corr2


def photoz_rsd_amplitude(background, z, k, z_obs_scatter, b_eff, mu):
    r"""Redshift-space halo amplitude with Gaussian photo-z damping.

    Computes the square-root amplitude whose square gives the anisotropic
    dispersion-model halo power-spectrum prefactor at fixed line-of-sight
    angle.

    Parameters
    ----------
    background : Background
        Background cosmology.
    z : float or np.ndarray
        Redshift.
    k : float or np.ndarray
        Wavenumber in :math:`h\,\mathrm{Mpc}^{-1}`.
    z_obs_scatter : float or np.ndarray
        Observed redshift scatter, :math:`\sigma_z`. If an array is
        provided, its first dimension must correspond to redshift.
    b_eff : float or np.ndarray
        Effective linear halo bias. If an array is provided, its first
        dimension must correspond to redshift.
    mu : float or np.ndarray
        Cosine of the angle between the wavevector and the line of sight.

    Returns
    -------
    np.ndarray
        Damped redshift-space halo amplitude,

        .. math::

            \left(b_{\mathrm{eff}} + f\mu^2\right)
            \exp\left[-\frac{1}{2}(k\sigma_r\mu)^2\right].

    Notes
    -----
    Squaring this amplitude gives the angular prefactor of the anisotropic
    power spectrum,

    .. math::

        P^s(k,\mu) =
        \left(b_{\mathrm{eff}}+f\mu^2\right)^2 P_m(k)
        \exp\left[-(k\sigma_r\mu)^2\right].
    """
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    z_obs_scatter_arr = np.asarray(z_obs_scatter)
    bias = np.asarray(b_eff)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100)
    )

    if z_obs_scatter_arr.ndim > 1:
        extra_axes = tuple(range(2, z_obs_scatter_arr.ndim + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)
    if z_obs_scatter_arr.ndim > 0:
        z_obs_scatter_arr = z_obs_scatter_arr[:, np.newaxis, ...]
        bias = bias[:, np.newaxis, ...]

    return (bias + f_gr * mu**2) * np.exp(-0.5 * (ks_z * z_obs_scatter_arr * mu) ** 2)


def tophat_window(kr):
    r"""Spherical top-hat window function.

    Parameters
    ----------
    kr : float or np.ndarray
        Dimensionless product of wavenumber and radius, :math:`kr`.

    Returns
    -------
    float or np.ndarray
        Spherical top-hat window,

        .. math::

            W_{\mathrm{th}}(x) =
            3\frac{\sin x - x\cos x}{x^3}.

    Notes
    -----
    For :math:`|kr|<0.1`, a Taylor expansion is used to avoid numerical
    cancellation around the origin.
    """
    kr = np.asarray(kr, dtype=float)
    window = np.empty_like(kr)
    small = np.abs(kr) < 0.1

    x_squared = kr[small] ** 2
    window[small] = 1.0 + x_squared * (
        -1.0 / 10.0
        + x_squared
        * (1.0 / 280.0 + x_squared * (-1.0 / 15120.0 + x_squared / 1330560.0))
    )

    x = kr[~small]
    window[~small] = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    return window


def tophat_window_derivative(kr):
    r"""Derivative of the spherical top-hat window function.

    Parameters
    ----------
    kr : float or np.ndarray
        Dimensionless product of wavenumber and radius, :math:`kr`.

    Returns
    -------
    float or np.ndarray
        Derivative :math:`dW_{\mathrm{th}}(kr)/d(kr)`.

    Notes
    -----
    The derivative is evaluated analytically as

    .. math::

        \frac{dW_{\mathrm{th}}}{dx} =
        3\frac{\sin x\,(x^2-3)+3x\cos x}{x^4}.
    """
    return 3.0 * (np.sin(kr) * (kr**2.0 - 3.0) + 3.0 * kr * np.cos(kr)) / kr**4.0


def isotropic_volume_distance(z, da, hz):
    r"""Isotropic volume-averaged distance.

    Parameters
    ----------
    z : float or np.ndarray
        Redshift.
    da : float or np.ndarray
        Angular-diameter distance.
    hz : float or np.ndarray
        Hubble parameter at ``z``.

    Returns
    -------
    float or np.ndarray
        Isotropic volume distance, with units inherited consistently from
        ``da``, ``hz``, and ``units.SPEED_OF_LIGHT``.

    Notes
    -----
    The distance is defined as

    .. math::

        D_V(z) =
        \left[(1+z)^2 D_A^2(z)\frac{cz}{H(z)}\right]^{1/3}.
    """
    return ((1 + z) ** 2 * da**2 * units.SPEED_OF_LIGHT * z / hz) ** (1 / 3.0)


def tabulated_return(reference_table, func, func_kwargs):
    r"""Return a cached function value or recompute it for new inputs.

    Checks whether the supplied keyword arguments are identical to the
    arguments stored in ``reference_table``. If they are, the cached value is
    returned. Otherwise, ``func`` is evaluated and both the inputs and output
    stored back in ``reference_table``.

    Parameters
    ----------
    reference_table : dict
        Mutable cache with an ``"inputs"`` dictionary and a ``"values"``
        entry. The keys of ``reference_table["inputs"]`` must match
        ``func_kwargs``.
    func : callable
        Function to evaluate when the cached inputs do not match.
    func_kwargs : dict
        Keyword arguments passed to ``func``.

    Returns
    -------
    Any
        Cached or newly computed value of ``func(**func_kwargs)``.

    Raises
    ------
    ValueError
        If the keys in ``func_kwargs`` do not match the keys in
        ``reference_table["inputs"]``.

    Notes
    -----
    When the inputs differ, ``reference_table`` is modified in place.
    """

    # Check if names of parameters given are the same as in reference_table
    _set_tab = set(reference_table["inputs"].keys())
    _set_inp = set(func_kwargs.keys())
    if _set_tab != _set_inp:
        raise ValueError(
            f"Bad parameters were passed. Expected {_set_tab}, got {_set_inp}."
        )

    # Check if func_kwargs are the tabluated values
    _tabuleted_input = True
    if any(value is None for value in reference_table["inputs"].values()):
        _tabuleted_input = False
    else:
        for name, ref_val in reference_table["inputs"].items():
            test_val = func_kwargs[name]
            if ref_val.shape != test_val.shape:
                _tabuleted_input = False
                break
            if (ref_val != test_val).any():
                _tabuleted_input = False
                break

    if not _tabuleted_input:
        reference_table["inputs"].update(func_kwargs)
        reference_table["values"] = func(**func_kwargs)

    return reference_table["values"]

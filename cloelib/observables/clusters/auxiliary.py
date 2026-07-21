"""Module implementing auxiliary functions"""

import numpy as np
from astropy import units as ap_units
from scipy.special import hyp1f1

from cloelib.auxiliary import units


def convert_to_Delta_crit(overdensity_type, overdensity=200, background=None, z=0.0):
    r"""Critical overdensity factor.

    Converts the input overdensity factor into a critical one.
    The contribution from massive neutrinos is not included in the matter density parameter.

    Parameters
    ----------
    overdensity_type : str
        Overdensity definition for halo mass calculation. Must be one of:
        - "crit": Relative to critical density of the universe.
        - "mean": Relative to mean matter density.
        - "vir": Virial overdensity from spherical collapse.
    overdensity : int, optional
        Value of the overdensity. Effective for non-virial overdensities.
        Example: If it equals 200, halos are defined as regions with density
        200 times the chosen reference (`crit` or `mean`).
    background : Background
        Background object.
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


def photoz_rsd_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the photo-z and RSD correction terms for the monopole.

    Parameters
    ----------
    background : Background
        Background cosmology.
    z : np.ndarray
        Redshift.
    k : np.ndarray
        Wavenumber in h Mpc^{-1}.
    z_obs_scatter : float, np.ndarray
        Observed redshift scatter. If array, its first dimension
        must correspond to redshift.

    Returns
    -------
    corr0, corr1, corr2 : np.ndarray
        Monopole correction terms with shape
        (z.size, k.size, ...).
    """
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    scatter = np.asarray(z_obs_scatter)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]

    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1.0e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100.0)
    )

    if scatter.ndim > 1:
        extra_axes = tuple(
            range(
                2,
                scatter.ndim + 1,
            )
        )

        f_gr = np.expand_dims(
            f_gr,
            axis=extra_axes,
        )

        ks_z = np.expand_dims(
            ks_z,
            axis=extra_axes,
        )

    if scatter.ndim > 0:
        scatter = scatter[
            :,
            np.newaxis,
            ...,
        ]

    x = (ks_z * scatter) ** 2

    moment0 = hyp1f1(
        0.5,
        1.5,
        -x,
    )

    moment1 = (
        hyp1f1(
            1.5,
            2.5,
            -x,
        )
        / 3.0
    )

    moment2 = (
        hyp1f1(
            2.5,
            3.5,
            -x,
        )
        / 5.0
    )

    corr0 = moment0
    corr1 = 2.0 * f_gr * moment1
    corr2 = f_gr**2 * moment2

    return corr0, corr1, corr2


def photoz_rsd_quadrupole_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the photo-z and RSD correction terms for the quadrupole."""
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    scatter = np.asarray(z_obs_scatter)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100)
    )

    if scatter.ndim > 1:
        extra_axes = tuple(range(2, scatter.ndim + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)
    if scatter.ndim > 0:
        scatter = scatter[:, np.newaxis, ...]

    x = (ks_z * scatter) ** 2
    moments = [hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1) for n in range(4)]

    corr0 = 2.5 * (3 * moments[1] - moments[0])
    corr1 = 5.0 * f_gr * (3 * moments[2] - moments[1])
    corr2 = 2.5 * f_gr**2 * (3 * moments[3] - moments[2])

    return corr0, corr1, corr2


def photoz_rsd_hexadecapole_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the photo-z and RSD correction terms for the hexadecapole."""
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    scatter = np.asarray(z_obs_scatter)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100)
    )

    if scatter.ndim > 1:
        extra_axes = tuple(range(2, scatter.ndim + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)
    if scatter.ndim > 0:
        scatter = scatter[:, np.newaxis, ...]

    x = (ks_z * scatter) ** 2
    moments = [hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1) for n in range(5)]

    corr0 = 9.0 / 8.0 * (35 * moments[2] - 30 * moments[1] + 3 * moments[0])
    corr1 = 9.0 / 4.0 * f_gr * (35 * moments[3] - 30 * moments[2] + 3 * moments[1])
    corr2 = 9.0 / 8.0 * f_gr**2 * (35 * moments[4] - 30 * moments[3] + 3 * moments[2])

    return corr0, corr1, corr2


def photoz_rsd_amplitude(background, z, k, z_obs_scatter, b_eff, mu):
    """Compute the redshift-space halo amplitude at fixed line-of-sight angle."""
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    scatter = np.asarray(z_obs_scatter)
    bias = np.asarray(b_eff)

    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * (units.SPEED_OF_LIGHT * 1e-3)
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100)
    )

    if scatter.ndim > 1:
        extra_axes = tuple(range(2, scatter.ndim + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)
    if scatter.ndim > 0:
        scatter = scatter[:, np.newaxis, ...]
        bias = bias[:, np.newaxis, ...]

    return (bias + f_gr * mu**2) * np.exp(-0.5 * (ks_z * scatter * mu) ** 2)


def tophat_window(kr):
    r"""compute top-hat window and its derivative.

    Parameters
    ----------
    kr: numpy.ndarray
           Wavenumber times radius.

    Returns
    -------
    numpy.ndarray
        Top-hat window function
    """
    return 3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr**3.0


def tophat_window_derivative(kr):
    r"""Compute derivative of the top-hat window.

    Parameters
    ----------
    kr: numpy.ndarray
           Wavenumber times radius.

    Returns
    -------
    numpy.ndarray
        Derivative of top-hat window function
    """
    return 3.0 * (np.sin(kr) * (kr**2.0 - 3.0) + 3.0 * kr * np.cos(kr)) / kr**4.0


def isotropic_volume_distance(z, da, hz):
    """Compute isotropic volume distance

    Parameters
    ----------
    z : np.ndarray
        redshift
    da : np.ndarray
        Angular diameter distance
    hz : np.ndarray
        Hubble parameter as a function of redshift.

    Returns
    -------
    np.ndarray
        Isotropic volume distance
    """
    return ((1 + z) ** 2 * da**2 * units.SPEED_OF_LIGHT * z / hz) ** (1 / 3.0)


def tabulated_return(reference_table, func, func_kwargs):
    """Check if arguments given are the same from the reference table.
    If true, returns the tabulated value. Otherwise, recomputes the output
    value and stores it (and inputs) back in reference table.

    Parameters
    ----------
    reference_table: dict
        Dictionary with the tabulated function. It must contain a `inputs`
        key with the dictionary of arguments or `function` and a `values`
        key with the computed value of said function with those arguments.
    func: function
        Function to be tabulated
    func_kwargs: dict
        Dictionary with named arguments for the function.

    Returns
    -------
    Output of `func(**func_kwargs)`
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

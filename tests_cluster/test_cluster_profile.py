# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_profile import BMOHaloProfile, NFWHaloProfile
from cloelib.observables.clusters.matter_statistics import MatterStatistics


def _get_matter_statistics():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    return MatterStatistics(perturbations)


def _get_castro():
    return CastroHaloAbundance(_get_matter_statistics())


def test_array_shapes():

    # Profiles
    print("# Profiles ")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
    )

    profile_nfw = NFWHaloProfile(_get_matter_statistics(), **_prof_kwargs)

    R_test = np.linspace(0.01, 1.0, 9)
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.linspace(1e14, 5e14, 6)
    c_test = 4.0
    HS = _get_matter_statistics()
    profile = NFWHaloProfile(HS)

    _kwargs = {"R": R_test, "z": z_test, "M": M_test}
    out_shape = (z_test.size, M_test.size, R_test.size)

    for radius_units in ("Mpc/h", "radians", "degrees", "arcmin", "arcsec"):
        print(radius_units)
        _kwargs["radius_units"] = radius_units

        _r, _z, _m = profile.core.surface_mass_density_args(**_kwargs)
        assert (_r * _z * _m).shape == out_shape

        _kwargs["c"] = c_test

        _kwargs["halo_bias"] = np.ones((z_test.size, M_test.size))

        for two_halo in ("None", "sum", "max"):
            profile_nfw.two_halo = two_halo

            assert profile_nfw.surface_mass_density(**_kwargs).shape == out_shape
            assert profile_nfw.excess_surface_mass_density(**_kwargs).shape == out_shape

        _kwargs.pop("halo_bias")
        _kwargs.pop("c")


def _test_profile(profile, reference_vals):

    R_test = np.array([1])
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.array([5e14])
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 5)
    mean_z_cl_test = np.linspace(0.2, 1.0, 4)
    HS = _get_matter_statistics()
    castro = _get_castro()
    halo_bias = castro.bias(z_test, M_test)

    print("    sigma_crit")
    assert_allclose(
        profile.core.sigma_crit(z_test, z_sources_test)[0],
        **reference_vals["sigma_crit"],
    )
    idx = np.array([profile.core._get_tomo_bin_index(mean_z_cl_test[i]) for i in range(mean_z_cl_test.size)])
    print("    n_zs_norM")
    n_zs_norMs = np.array([profile.core.n_zs_norM(mean_z_cl_test[i],idx[i]) for i in range(mean_z_cl_test.size)])
    assert_allclose(n_zs_norMs, **reference_vals["n_zs_norM"],atol=1.0e-5)
    print("    n_zs")
    nzs = np.array([profile.core.n_zs(z_sources_test,idx[i]) for i in range(mean_z_cl_test.size)])
    assert_allclose(nzs, **reference_vals["n_zs"],atol=1.0e-5)
    print("    surface_mass_density")
    assert_allclose(
        profile.surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["surface_mass_density"],
    )
    print("    excess_surface_mass_density")
    assert_allclose(
        profile.excess_surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["excess_surface_mass_density"],
    )
    print("    _surface_mass_density_2h")
    profile_2h = (
        profile.core.matter_statistics.surface_mass_density_2h(R_test, z_test)[
            :, np.newaxis, :
        ]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["surface_mass_density_2h"],
    )
    print("    _excess_surface_mass_density_2h")
    profile_2h = (
        profile.core.matter_statistics.excess_surface_mass_density_2h(R_test, z_test)[
            :, np.newaxis, :
        ]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["excess_surface_mass_density_2h"],
    )


def test_profiles():

    # Profiles
    print("# Profiles ")

    print("  NFW")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
    )
    profile_nfw = NFWHaloProfile(_get_matter_statistics(), **_prof_kwargs)
    _reference_vals = {
        # All validation values have to be updated with extarnal values
        "sigma_crit": {
            "desired": [
                57269.705861,
                57134.890066,
                57034.779821,
                56957.599339,
                56896.34007,
            ],
            "rtol": 1e-5,
        },
        "n_zs_norM": {"desired": [1.12513478, 1.50243006, 1.28488508, 1.594103  ], "rtol": 1e-5},
        "n_zs": {
            "desired": [[0.83347014, 0.7820134 , 0.72356751, 0.66030692, 0.59431415],
       [0.83347014, 0.7820134 , 0.72356751, 0.66030692, 0.59431415],
       [0.29190244, 0.5992073 , 0.84143889, 0.92745609, 0.9132132 ],
       [0.08161356, 0.18759727, 0.34873945, 0.53502114, 0.69604214]],
            "rtol": 1e-5,
        },
        "surface_mass_density": {
            "desired": [57.782346, 60.278322, 62.62685, 64.791899],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density": {
            "desired": [85.397604, 94.3122, 103.483482, 112.700292],
            "rtol": 1e-3,
        },
        "surface_mass_density_2h": {
            "desired": [13.316477, 16.810355, 20.655992, 24.778575],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density_2h": {
            "desired": [1.520876, 2.16498, 2.853623, 3.655179],
            "rtol": 1e-3,
        },
    }
    _test_profile(profile_nfw, _reference_vals)

    print("  BMO")
    _reference_vals.update(
        # All validation values have to be updated with extarnal values
        {
            "surface_mass_density": {
                "desired": [49.754672, 50.309348, 50.479286, 50.276593],
                "rtol": 1e-3,
            },
            "excess_surface_mass_density": {
                "desired": [90.134602, 99.25608, 108.537932, 117.745383],
                "rtol": 1e-3,
            },
        }
    )

    _prof_kwargs = dict(
        two_halo="None",
        trunc_fact=3.0,
        zs_max=2.0,
    )
    profile_bmo = BMOHaloProfile(_get_matter_statistics(), **_prof_kwargs)
    _test_profile(profile_bmo, _reference_vals)

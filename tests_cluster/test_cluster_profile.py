# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_profile import BMOHaloProfile, NFWHaloProfile
from cloelib.observables.clusters.halo_profile import EmulatorMiscenteredHaloProfile
from cloelib.observables.clusters.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics
from cloelib.observables.clusters.selection_function import GaussianSelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterStatisticsModeling,
    ClusterWeakLensing,
    ClusterWeakLensingCovariance,
)


def _get_matter_statistics(**kwargs):
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

    return MatterStatistics(perturbations, **kwargs)


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

def _test_misc_profile(profile, reference_vals):
    r"""
    Test miscentered Sigma_off and DeltaSigma_off profiles against benchmark
    values.
 
    Parameters
    ----------
    profile : MiscBMOHaloProfileEmu
        Instantiated emulator profile object.
    reference_vals : dict
        Dictionary with keys ``"surface_mass_density"`` and
        ``"excess_surface_mass_density"``, each containing ``"desired"``
        (h Msun / pc²) and ``"rtol"``.
    """
    # Input grid matching the benchmark generation
    R_test = 10.0 ** np.arange(2.0, 4.5, 0.05) / 1.0e3  # physical Mpc/h, (50,)
    z_test = np.array([1.0])                               # (1,)
    M_test = np.array([1.0e14])                            # Msun/h, (1,)
    c_test = 2.0
    sigma_off_test = 0.4                                   # physical Mpc/h
 
    out_shape = (z_test.size, M_test.size, R_test.size)   # (1, 1, 50)
 
    print("    surface_mass_density (misc)")
    Sigma_off = profile.surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    assert Sigma_off.shape == out_shape, (
        f"surface_mass_density shape mismatch: got {Sigma_off.shape}, "
        f"expected {out_shape}"
    )
    assert_allclose(Sigma_off[0, 0, :], **reference_vals["surface_mass_density"])
 
    print("    excess_surface_mass_density (misc)")
    DeltaSigma_off = profile.excess_surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    assert DeltaSigma_off.shape == out_shape, (
        f"excess_surface_mass_density shape mismatch: got {DeltaSigma_off.shape}, "
        f"expected {out_shape}"
    )
    assert_allclose(
        DeltaSigma_off[0, 0, :], **reference_vals["excess_surface_mass_density"]
    )
 
 
def test_misc_profile():
    r"""
    Integration test for MiscBMOHaloProfileEmu.
 
    Benchmark arrays were produced at:
        M = 1e14 Msun/h,  z = 1.0,  c = 2.0,  sigma_off = 0.4 Mpc/h,
        H0 = 67.7,  omch2 = 0.120,  ombh2 = 0.022
        R = 10^arange(2, 4.5, 0.05) / 1e3  physical Mpc/h.
 
    The benchmark values are in units of Msun h / physical Mpc², which
    convert to h Msun / pc² via the factor 1e-12
    (1 Mpc = 1e6 pc  →  1/Mpc² = 1e-12/pc²).
 
    Weights are loaded automatically from EmuNetWeights (no path arguments
    required).
    """
    print("# MiscBMOHaloProfileEmu")
 
    # Benchmark in Msun h / physical Mpc²; convert to h Msun / pc² via 1e-12.
    _sigma_off_mpc2 = np.array([
        1.03437782e+14, 1.02892023e+14, 1.02208253e+14, 1.01353450e+14,
        1.00287558e+14, 9.89624947e+13, 9.73213312e+13, 9.52978910e+13,
        9.28171457e+13, 8.97969655e+13, 8.61519774e+13, 8.18004608e+13,
        7.66752572e+13, 7.07393939e+13, 6.40062278e+13, 5.65620751e+13,
        4.85863345e+13, 4.03605128e+13, 3.22550638e+13, 2.46848866e+13,
        1.80343758e+13, 1.25710012e+13, 8.38335067e+12, 5.37794615e+12,
        3.33890203e+12, 2.01484296e+12, 1.18579349e+12, 6.81083267e+11,
        3.82348622e+11, 2.10344854e+11, 1.13748288e+11, 6.06457586e+10,
        3.19638895e+10, 1.66916789e+10, 8.65202212e+09, 4.45809141e+09,
        2.28611968e+09, 1.16779650e+09, 5.94661984e+08, 3.02036626e+08,
        1.53086286e+08, 7.74570941e+07, 3.91348831e+07, 1.97492122e+07,
        9.95643275e+06, 5.01528637e+06, 2.52454382e+06, 1.27002555e+06,
        6.38591907e+05, 3.20958012e+05,
    ])
 
    _dsigma_off_mpc2 = np.array([
        1.21628769e+12, 1.49424755e+12, 1.84603967e+12, 2.26596256e+12,
        2.79707329e+12, 3.44687095e+12, 4.25073964e+12, 5.24109630e+12,
        6.42031402e+12, 7.83788536e+12, 9.52616609e+12, 1.14948945e+13,
        1.37443295e+13, 1.62413303e+13, 1.89310467e+13, 2.17273282e+13,
        2.44462940e+13, 2.68214658e+13, 2.84883561e+13, 2.93805054e+13,
        2.92776593e+13, 2.81449114e+13, 2.60604369e+13, 2.33498010e+13,
        2.03703578e+13, 1.73444167e+13, 1.45191592e+13, 1.19739835e+13,
        9.78197961e+12, 7.91753709e+12, 6.37809774e+12, 5.11051992e+12,
        4.08501406e+12, 3.25849912e+12, 2.59579994e+12, 2.06687228e+12,
        1.64387867e+12, 1.30401149e+12, 1.04122801e+12, 8.23317572e+11,
        6.55088494e+11, 5.19934196e+11, 4.13581168e+11, 3.28310063e+11,
        2.60977427e+11, 2.07169421e+11, 1.64662861e+11, 1.30755777e+11,
        1.03861854e+11, 8.25733749e+10,
    ])
 
    _reference_vals = {
        "surface_mass_density": {
            "desired": _sigma_off_mpc2 * 1.0e-12,   # h Msun / pc²
            "rtol": 1e-2,                            # emulator-level accuracy (~1%)
        },
        "excess_surface_mass_density": {
            "desired": _dsigma_off_mpc2 * 1.0e-12,  # h Msun / pc²
            "rtol": 1.5e-2,
        },
    }
 
    _prof_kwargs = dict(
        zs_max=2.0,
    )
    profile_misc = EmulatorMiscenteredHaloProfile(_get_matter_statistics(), **_prof_kwargs)
    profile_misc.set_weights()
    _test_misc_profile(profile_misc, _reference_vals)


def test_gt_covariance():

    # Gaussian covariance of the reduced shear profile
    print("# g_t covariance")

    integ_ztrue_arr = np.linspace(0.1, 1.1, 101)
    integ_mass_arr = np.logspace(12.0, 16.0, 121)
    matter_stat = _get_matter_statistics(
        z=np.linspace(0.01, 2.0, 100), k=np.geomspace(1e-4, 10.0, 500)
    )
    selection_function = GaussianSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0, B_l=0.9, C_l=0.5, sig_A_l=0.2, sig_B_l=-0.05, sig_C_l=0.001
        ),
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
        lambda_tab_integ=[31, 31],
        z_tab_integ=31,
    )
    cluster_statistics_modeling = ClusterStatisticsModeling(
        CastroHaloAbundance(matter_statistics=matter_stat),
        selection_function,
        integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
        integ_mass_arr=integ_mass_arr,
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
        integ_ztrue_arr=integ_ztrue_arr,
        area=10313.0,
    )
    cluster_wl = ClusterWeakLensing(
        cluster_statistics_modeling,
        NFWHaloProfile(matter_stat, two_halo="None", zs_max=2.0),
        halo_concentration=4.0,
    )
    cluster_wl_cov = ClusterWeakLensingCovariance(cluster_wl, n_eff=15.0, sigma_e=0.3)

    z_obs_edges = np.array([0.3, 0.5, 0.7])
    lambda_obs_edges = np.array([20.0, 30.0, 60.0])
    radius_edges = np.geomspace(0.3, 10.0, 6)
    z_size, lambda_size, radius_size = 2, 2, 5

    cov, terms = cluster_wl_cov.get_gt_covariance(
        z_obs_edges, lambda_obs_edges, radius_edges, return_terms=True
    )

    print("    shape and symmetries")
    assert cov.shape == (z_size, z_size, lambda_size, lambda_size) + (radius_size,) * 2
    assert_allclose(cov, terms["shape"] + terms["lss"] + terms["intr"], rtol=1e-12)
    # diagonal in z_obs bins
    assert_equal(cov[0, 1], 0.0)
    assert_equal(cov[1, 0], 0.0)
    # symmetric and positive definite
    cov_2d = cov.transpose(0, 2, 4, 1, 3, 5).reshape(
        (z_size * lambda_size * radius_size,) * 2
    )
    assert_allclose(cov_2d, cov_2d.T, rtol=1e-12)
    assert np.linalg.eigvalsh(cov_2d).min() > 0.0

    print("    auto richness blocks")
    cov_auto = cluster_wl_cov.get_gt_covariance(
        z_obs_edges, lambda_obs_edges, radius_edges, cross_richness=False
    )
    for ind_lambda in range(lambda_size):
        assert_allclose(
            cov_auto[:, :, ind_lambda, ind_lambda],
            cov[:, :, ind_lambda, ind_lambda],
            rtol=1e-12,
        )
    assert_equal(cov_auto[:, :, 0, 1], 0.0)
    assert np.abs(cov[:, :, 0, 1]).max() > 0.0

    # Reference values from the notebook "Test Covariance lensing gammat.ipynb"
    # (last cell, cl_157_analensing_cov), computed with the original implementation of Wu et al. 2019
    # Eq. 10 using the same cloelib ingredients.
    # Auto richness blocks, shape (z_obs, lambda_obs, radius, radius).
    # The variances are compared with relative tolerance, the off-diagonal
    # terms through the correlation matrix: the notebook integrates numerically
    # the lens shot noise x shape noise term, which leaks a spurious correlation
    # between adjacent radial bins of ~1e-4 (exactly zero analytically).
    print("    comparison with external values")
    cov_reference = np.array(
        [
            [
                [
                    [3.281672e-08, 7.077090e-11, 1.568597e-11, 4.359248e-12, 1.012907e-12],
                    [7.077090e-11, 8.189872e-09, 6.449401e-11, 1.706008e-11, 4.310390e-12],
                    [1.568597e-11, 6.449401e-11, 2.130659e-09, 6.830604e-11, 1.749422e-11],
                    [4.359248e-12, 1.706008e-11, 6.830604e-11, 6.478853e-10, 7.045661e-11],
                    [1.012907e-12, 4.310390e-12, 1.749422e-11, 7.045661e-11, 2.848028e-10],
                ],
                [
                    [5.820921e-08, 1.349009e-10, 2.970522e-11, 7.136410e-12, 1.878316e-12],
                    [1.349009e-10, 1.454609e-08, 1.229641e-10, 3.036858e-11, 7.509319e-12],
                    [2.970522e-11, 1.229641e-10, 3.795613e-09, 1.232244e-10, 3.082744e-11],
                    [7.136410e-12, 3.036858e-11, 1.232244e-10, 1.155562e-09, 1.247561e-10],
                    [1.878316e-12, 7.509319e-12, 3.082744e-11, 1.247561e-10, 5.054527e-10],
                ],
            ],
            [
                [
                    [3.260651e-08, 6.634224e-11, 1.543328e-11, 4.217002e-12, 1.229022e-12],
                    [6.634224e-11, 8.129639e-09, 6.326760e-11, 1.776314e-11, 4.561915e-12],
                    [1.543328e-11, 6.326760e-11, 2.115162e-09, 7.072966e-11, 1.869162e-11],
                    [4.217002e-12, 1.776314e-11, 7.072966e-11, 6.493981e-10, 7.505699e-11],
                    [1.229022e-12, 4.561915e-12, 1.869162e-11, 7.505699e-11, 2.940720e-10],
                ],
                [
                    [6.404261e-08, 1.350488e-10, 3.092210e-11, 8.222356e-12, 2.129840e-12],
                    [1.350488e-10, 1.597659e-08, 1.271718e-10, 3.471310e-11, 8.749552e-12],
                    [3.092210e-11, 1.271718e-10, 4.160022e-09, 1.383706e-10, 3.592380e-11],
                    [8.222356e-12, 3.471310e-11, 1.383706e-10, 1.274928e-09, 1.445602e-10],
                    [2.129840e-12, 8.749552e-12, 3.592380e-11, 1.445602e-10, 5.709630e-10],
                ],
            ],
        ]
    )
    cov_auto_blocks = np.array(
        [
            [cov[iz, iz, il, il] for il in range(lambda_size)]
            for iz in range(z_size)
        ]
    )
    variance = np.diagonal(cov_auto_blocks, axis1=2, axis2=3)
    variance_reference = np.diagonal(cov_reference, axis1=2, axis2=3)
    assert_allclose(variance, variance_reference, rtol=1e-2)

    correlation = cov_auto_blocks / np.sqrt(
        variance[..., :, np.newaxis] * variance[..., np.newaxis, :]
    )
    correlation_reference = cov_reference / np.sqrt(
        variance_reference[..., :, np.newaxis] * variance_reference[..., np.newaxis, :]
    )
    assert_allclose(correlation, correlation_reference, atol=2e-3)

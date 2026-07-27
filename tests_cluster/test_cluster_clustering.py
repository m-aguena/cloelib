# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.auxiliary import (
    photoz_rsd_amplitude,
    photoz_rsd_monopole_correction,
    photoz_rsd_hexadecapole_correction,
    photoz_rsd_quadrupole_correction,
)
from cloelib.observables.clusters.halo_clustering import TwoPoint3DHaloClustering
from cloelib.observables.clusters.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


def _test_clustering(CL, perturbations):
    z_test = np.array([0.0, 1.0])
    r_test = np.array([30.0, 60.0, 90.0])
    lob_test = np.array([50.0])
    k_test = np.geomspace(1e-4, 10, 500)

    print("    alcock_paczynski_correction_factor")
    ref_APcorr = np.array([1.0162, 1.016033])
    assert_allclose(
        CL.core.alcock_paczynski_correction_factor(z_test), ref_APcorr, rtol=1e-04
    )

    print("    radial_shell_window_and_volume")
    ref_radial_shell_window_and_volume0 = np.array(
        [
            [[9.99995884e-01, -1.35023865e-05], [9.99989679e-01, 8.02679298e-06]],
            [[9.99995885e-01, -1.37175506e-05], [9.99989682e-01, 8.32722698e-06]],
        ]
    )

    ref_radial_shell_window_and_volume1 = np.array(
        [[830784.512318, 2254986.533435], [830373.221984, 2253870.173956]]
    )

    WF, VF = CL.core.radial_shell_window_and_volume(z_test, k_test, r_test)
    assert_allclose(WF[:, :, [0, -1]], ref_radial_shell_window_and_volume0, rtol=1e-03)
    assert_allclose(VF, ref_radial_shell_window_and_volume1, rtol=1e-03)

    WF2, VF2 = CL.core.radial_shell_quadrupole_window_and_volume(
        z_test, k_test, r_test
    )
    WF4, VF4 = CL.core.radial_shell_hexadecapole_window_and_volume(
        z_test, k_test, r_test
    )
    assert_equal(WF2.shape, WF.shape)
    assert_equal(WF4.shape, WF.shape)
    assert_allclose(VF2, VF)
    assert_allclose(VF4, VF)

    print("    Pk_IR_func")
    ref_Pk_IR = np.array([[4.228415e02, 1.061383e-01], [1.561681e02, 3.934860e-02]])

    Pk_test = perturbations.matter_power_spectrum(
        z_test, k_test, hubble_units=True, k_hunit=True
    )
    assert_allclose(
        CL.core.Pk_IR_func(k_test, Pk_test)[:, [0, -1]], ref_Pk_IR, rtol=1e-4
    )


def test_clustering():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _ns = 0.96
    _mnu = 0.06
    _As = 2.0e-9
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=_ns,
        mnu=_mnu,
        As=_As,
        gamma_MG=0.0,
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    matter_statistics = MatterStatistics(
        perturbations,
        # z=integ_ztrue_arr,
        k=np.geomspace(1e-4, 10, 500),
    )

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    CL = TwoPoint3DHaloClustering(matter_statistics, background_fid)
    _test_clustering(CL, perturbations)


def test_cosmo_photoz_rsd_correction():

    print("# Cosmology parameters")
    _cosmo_pars = dict(
        H0=67.7,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_b0=0.022 / 0.677**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    z_test = np.array([0.0, 1.0])
    lob_test = np.array([50.0])
    k_test = np.geomspace(1e-4, 10, 500)
    zobs_scatter = np.array([5.0, 5.1])

    background = CAMBBackground(**_cosmo_pars)

    corr0, corr1, corr2 = photoz_rsd_monopole_correction(
        background,
        z_test,
        k_test,
        zobs_scatter,
    )

    # test values

    ref_phz_rsd_0 = np.array(
        [[5.7111615e-01, 5.9122967e-06], [8.0073649e-01, 1.0336005e-05]]
    )
    ref_phz_rsd_1 = np.array(
        [[1.087285e-01, 1.381287e-16], [3.810938e-01, 1.225914e-15]]
    )
    ref_phz_rsd_2 = np.array(
        [[1.256838e-02, 2.420331e-27], [9.109210e-02, 1.090508e-25]]
    )

    assert_allclose(corr0[:, [0, -1]], ref_phz_rsd_0, rtol=1e-04)
    assert_allclose(corr1[:, [0, -1]], ref_phz_rsd_1, rtol=1e-04)
    assert_allclose(corr2[:, [0, -1]], ref_phz_rsd_2, rtol=1e-04)


def test_cosmo_photoz_rsd_multipoles():
    cosmo_pars = dict(
        H0=67.7,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_b0=0.022 / 0.677**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )
    background = CAMBBackground(**cosmo_pars)
    z = np.array([0.2, 0.8])
    k = np.geomspace(1e-3, 1.0, 12)
    bias = np.array([2.0, 3.0])

    corr2 = photoz_rsd_quadrupole_correction(
        background, z, k, np.zeros_like(z)
    )
    corr4 = photoz_rsd_hexadecapole_correction(
        background, z, k, np.zeros_like(z)
    )
    pk2_over_pm = corr2[0] * bias[:, None] ** 2 + corr2[1] * bias[:, None] + corr2[2]
    pk4_over_pm = corr4[0] * bias[:, None] ** 2 + corr4[1] * bias[:, None] + corr4[2]

    growth_rate = background.Omega_cb(z) ** 0.55
    expected2 = (
        4.0 / 3.0 * bias * growth_rate + 4.0 / 7.0 * growth_rate**2
    )[:, None]
    expected4 = (8.0 / 35.0 * growth_rate**2)[:, None]
    assert_allclose(pk2_over_pm, np.broadcast_to(expected2, pk2_over_pm.shape))
    assert_allclose(pk4_over_pm, np.broadcast_to(expected4, pk4_over_pm.shape))

    scatter = np.array([0.01, 0.02])
    corr2 = photoz_rsd_quadrupole_correction(background, z, k, scatter)
    corr4 = photoz_rsd_hexadecapole_correction(background, z, k, scatter)
    analytic2 = corr2[0] * bias[:, None] ** 2 + corr2[1] * bias[:, None] + corr2[2]
    analytic4 = corr4[0] * bias[:, None] ** 2 + corr4[1] * bias[:, None] + corr4[2]

    mu, weights = np.polynomial.legendre.leggauss(1024)
    direct2 = np.zeros_like(analytic2)
    direct4 = np.zeros_like(analytic4)
    for mu_i, weight in zip(mu, weights):
        amplitude = photoz_rsd_amplitude(
            background, z, k, scatter, bias, mu_i
        )
        direct2 += 2.5 * weight * 0.5 * (3.0 * mu_i**2 - 1.0) * amplitude**2
        direct4 += (
            4.5
            * weight
            * (35.0 * mu_i**4 - 30.0 * mu_i**2 + 3.0)
            / 8.0
            * amplitude**2
        )

    assert_allclose(analytic2, direct2, rtol=1e-9, atol=1e-11)
    assert_allclose(analytic4, direct4, rtol=1e-9, atol=1e-11)
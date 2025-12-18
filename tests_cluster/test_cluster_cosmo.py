# impoer jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)


def test_cosmo():
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
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    # background
    background = CAMBBackground(**_cosmo_pars)
    assert background.Omega_m_cb(0) < background.Omega_m(0)
    assert_allclose(
        background.Omega_m_cb(0),
        _cosmo_pars["Omega_cdm0"] + _cosmo_pars["Omega_b0"],
        rtol=1e-03,
    )
    assert_allclose(
        derived_cosmology.rho_crit(background, 0), 1.27203085e11, rtol=1e-03
    )
    assert_allclose(background.rdrag, 147.50225, rtol=1e-05)

    # camb linear
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    assert_allclose(perturbations.matter_power_spectrum(0, 1), 80.534861)
    assert_allclose(perturbations.matter_power_spectrum_cb(0, 1), 81.748209, rtol=1e-03)

    # camb non-linear
    perturbations_nl = CAMBNonLinearPerturbations(
        background, np.linspace(0.0, 2.0, 100)
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum(0, 1), 736.010737, rtol=1.0e-03
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum_cb(0, 1), 747.017036, rtol=1.0e-03
    )


def test_cosmo_photoz_rsd_correction():

    from cloelib.observables.clusters.selection_function import SelectionFunction

    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _ns = 0.96
    _mnu = 0.06
    _As = 2.0e-9

    background = CAMBBackground(
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
    SF = SelectionFunction(
        A_l=0.5,
        B_l=0.6,
        C_l=0.5,
        sig_A_l=0.1,
        sig_B_l=0.0,
        sig_C_l=0.0,
        sig_lambda_norm=0.1,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.1,
        sig_z_z=0.1,
        sig_z_lambda=0.1,
    )

    z_test = np.array([0.0, 1.0])
    lob_test = np.array([50.0])
    k_test = np.geomspace(1e-4, 10, 500)
    corr0, corr1, corr2 = derived_cosmology.photoz_rsd_correction(
        background,
        z_test,
        k_test,
        SF.scatter_zobs_z(lob_test, z_test),
        nonu=True,
    )

    # test values

    ref_phz_rsd_0 = np.array(
        [[5.7111615e-01, 5.9122967e-06], [8.0073649e-01, 1.0336005e-05]]
    )
    ref_phz_rsd_1 = np.array(
        [[1.0873061e-01, 1.3813213e-16], [3.8109362e-01, 1.2259151e-15]]
    )
    ref_phz_rsd_2 = np.array(
        [[1.2568793e-02, 2.4204413e-27], [9.1092102e-02, 1.0905091e-25]]
    )

    assert_allclose(corr0[:, [0, -1]], ref_phz_rsd_0, rtol=1e-04)
    assert_allclose(corr1[:, [0, -1]], ref_phz_rsd_1, rtol=1e-04)
    assert_allclose(corr2[:, [0, -1]], ref_phz_rsd_2, rtol=1e-04)

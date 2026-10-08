# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_abundance import CastroHaloAbundance
from cloelib.observables.halos.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.selection_function import GaussianSelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterCounts,
    ClusterStatisticsModeling,
    Covariance,
)


def test_count_covariance():
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

    # counts covariance
    print("# counts covariance")

    area = 15000
    nbins_z = 10
    z_tab_integ = 31

    k_min = 1e-4
    k_max = 2e0
    k_div = 300

    zbins = np.linspace(0, 2, nbins_z + 1)
    k_test = np.geomspace(k_min, k_max, k_div)

    halo_model_properties = HaloModelProperties(
        perturbations,
        k=np.geomspace(k_min / 2, k_max * 2, k_div),
    )
    halo_abundance = CastroHaloAbundance(halo_model_properties)
    selection_function = GaussianSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
        z_tab_integ=z_tab_integ,
        lambda_tab_integ=[31],
    )
    clusterstatmod = ClusterStatisticsModeling(
        halo_abundance=halo_abundance,
        selection_function=selection_function,
        integ_k_arr=k_test,
        integ_mass_arr=np.geomspace(1e12, 1e16, 31),
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 31),
        integ_ztrue_arr=np.linspace(0.01, 1.99, 31),
        area=area,
    )
    CC = Covariance(ClusterCounts(clusterstatmod), z_tab_integ=z_tab_integ)
    # Exercise the unchanged internal survey-window kernel against reference values.
    CC.rint = np.zeros((nbins_z, len(k_test), CC.L + 1))

    print("    Covariance coefficients")
    KL = CC._Kl_coeff()
    # All validation values have to be updated with extarnal values
    assert_allclose(
        KL[:5], [0.282095, 0.310942, 0.1095, -0.074565, -0.091053], rtol=5e-6
    )

    print("    Covariance window")
    iz = 0
    # All validation values have to be updated with extarnal values
    assert_allclose(
        CC._cov_window(iz, zbins[iz : iz + 2], KL)[0, :5],
        np.array([0.99959593, 0.99956826, 0.9995387, 0.99950711, 0.99947336]),
        rtol=1e-6,
    )

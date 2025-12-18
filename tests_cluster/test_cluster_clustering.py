# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.clustering import HaloClustering


def _test_clustering(CL, perturbations):
    z_test = np.array([0.0, 1.0])
    r_test = np.array([30.0, 60.0, 90.0])
    lob_test = np.array([50.0])

    print("    APcorr_func")
    ref_APcorr = np.array([1.0162, 1.016033])
    assert_allclose(CL.APcorr_func(z_test), ref_APcorr, rtol=1e-04)

    print("    WF_ra")
    ref_WF_ra0 = np.array(
        [
            [[9.99995884e-01, -1.35023865e-05], [9.99989679e-01, 8.02679298e-06]],
            [[9.99995885e-01, -1.37175506e-05], [9.99989682e-01, 8.32722698e-06]],
        ]
    )

    ref_WF_ra1 = np.array(
        [[830784.512318, 2254986.533435], [830373.221984, 2253870.173956]]
    )

    WF, VF = CL.WF_ra(z_test, r_test)
    assert_allclose(WF[:, :, [0, -1]], ref_WF_ra0, rtol=1e-03)
    assert_allclose(VF, ref_WF_ra1, rtol=1e-03)

    print("    Pk_IR_func")
    ref_Pk_IR = np.array([[4.2284186e02, 1.0611498e-01], [1.5616818e02, 3.9339960e-02]])

    Pk_test = perturbations.matter_power_spectrum(
        z_test, CL.k, hubble_units=True, k_hunit=True
    )
    assert_allclose(CL.Pk_IR_func(Pk_test)[:, [0, -1]], ref_Pk_IR, rtol=1e-4)


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

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )
    CL = HaloClustering(perturbations, perturbations_fid, nonu=True)
    _test_clustering(CL, perturbations)

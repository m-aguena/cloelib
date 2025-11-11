# import jax.numpy as np
import benchmark_values
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.castro_hmf_bias import CastroHMFBias
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.profile import ProfileNFW
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterCounts,
    ClusterWL,
    ClusterXi2,
)


def get_values():

    ###########
    # Cosmology
    ###########

    print("# Cosmology parameters")
    _H0 = 67.0
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
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )

    #############
    # Observables
    #############

    # Parameters

    _sel_pars = dict(
        A_l=52.0,
        B_l=0.9,
        C_l=0.5,
        sig_A_l=0.2,
        sig_B_l=-0.05,
        sig_C_l=0.001,
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
    )

    _prof_pars = dict(
        r_interp=np.logspace(-10, 2.5, 200),
        two_halo="None",
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )

    integ_k_arr = np.geomspace(1e-4, 10, 500)
    integ_mass_arr = np.logspace(12.0, 16.0, 51)
    integ_lambda_true_arr = np.geomspace(5.0, 250.0, 51)
    integ_ztrue_arr = np.linspace(1.0e-5, 6.0 - 1.0e-5, 200)

    halo_concentration = 0.1
    overdensity_type = "vir"
    area = 10313

    # Integration bins

    zed_obs_nc_bins = np.linspace(0.2, 1.8, 9)
    lambda_obs_nc_bins = np.array([20.0, 30.0, 45.0, 60.0, 500.0])
    radius_profile_bins = np.linspace(5.0, 100.0, 11)
    lambda_obs_xi2_bins = np.array([20, 30, 500])
    radius_xi2_bins = np.geomspace(20.0, 130.0, 31)
    zed_obs_xi2_bins = np.arange(0.2, 1.81, 0.4)

    # Istanciate objects

    selectionFunction = SelectionFunction(**_sel_pars)
    HSCastro = CastroHMFBias(
        halo_statistics=HaloStatistics(
            perturbations,
            z=integ_ztrue_arr,
            k=integ_k_arr,
            overdensity_type=overdensity_type,
        )
    )
    covariance = HaloCovariance(
        perturbations, area=area, nbins_zob=len(zed_obs_nc_bins), k=integ_k_arr
    )
    profileNFW = ProfileNFW(HSCastro, k=integ_k_arr, z=integ_ztrue_arr, **_prof_pars)
    haloClustering = HaloClustering(
        perturbations, perturbations_fid, selectionFunction, k=integ_k_arr
    )

    ####################
    # Summary Statistics
    ####################

    # Istanciate objects

    cluster_counts_statistics = ClusterCounts(
        HSCastro,
        selectionFunction,
        covariance,
        integ_k_arr=integ_k_arr,
        integ_mass_arr=integ_mass_arr,
        integ_lambda_true_arr=integ_lambda_true_arr,
        integ_ztrue_arr=integ_ztrue_arr,
        area=area,
        photoz_rsd_correction=haloClustering.photoz_rsd_correction,
    )
    cluster_wl_statistics = ClusterWL(
        cluster_counts_statistics,
        profileNFW,
        halo_concentration=halo_concentration,
    )
    cluster_xi2_statistics = ClusterXi2(
        cluster_counts_statistics,
        haloClustering,
        area=area,
    )

    # Compute values

    nc_zbin_lbin, counts_aux = cluster_counts_statistics.compute_binned_quantities(
        z_obs_bins=zed_obs_nc_bins,
        lambda_obs_bins=lambda_obs_nc_bins,
    )
    cov_nc_zbin_lbin = cluster_counts_statistics.compute_cov(
        zed_obs_nc_bins, nc_zbin_lbin, counts_aux["Plob_M_z"], counts_aux["dV_dzob"]
    )
    gt_zbin_lbin_rbin = cluster_wl_statistics.compute_binned_quantities(
        z_obs_bins=zed_obs_nc_bins,
        lambda_obs_bins=lambda_obs_nc_bins,
        radius_bins=radius_profile_bins,
    )
    xi2_zbin_lbin_rbin, xi2_aux = cluster_xi2_statistics.compute_binned_quantities(
        lambda_obs_bins=lambda_obs_xi2_bins,
        radius_bins=radius_xi2_bins,
        z_obs_bins=zed_obs_xi2_bins,
    )
    cov_xi2_zbin_lbin_rbin = cluster_xi2_statistics.compute_cov(
        xi2_aux["Pk_lambdai_lambdaj"],
        xi2_aux["one_over_n_lambdai_lambdaj"],
        xi2_aux["shell_window"],
        xi2_aux["shell_volume"],
        xi2_aux["volume_zob"],
    )
    return (
        nc_zbin_lbin,
        gt_zbin_lbin_rbin,
        xi2_zbin_lbin_rbin,
        cov_nc_zbin_lbin,
        cov_xi2_zbin_lbin_rbin,
    )


def test_clustersummmarystatitistics():
    (
        nc_zbin_lbin,
        gt_zbin_lbin_rbin,
        xi2_zbin_lbin_rbin,
        cov_nc_zbin_lbin,
        cov_xi2_zbin_lbin_rbin,
    ) = get_values()

    assert_allclose(nc_zbin_lbin, benchmark_values.nc_ref, rtol=1e-2)

    assert_allclose(gt_zbin_lbin_rbin[0:2], benchmark_values.gt, rtol=1e-2)

    assert_allclose(xi2_zbin_lbin_rbin[0:2], benchmark_values.xi2, rtol=1e-2)

    assert_allclose(cov_nc_zbin_lbin[1:2], benchmark_values.nc_cov, rtol=5e-2)

    assert_allclose(
        cov_xi2_zbin_lbin_rbin[1, 1, 1:3, 1:3, 10:20, 10:20],
        benchmark_values.xi2_cov,
        rtol=5e-2,
    )


if __name__ == "__main__":
    get_values()

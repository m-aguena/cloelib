"""

## Notes :

- Cluster profile lensing

"""

# General imports
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import j0, j1, sici

# cloelib imports
from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_profile import HaloProfile
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax


class ClusterWeakLensing:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        profile: HaloProfile,
        halo_concentration: float,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        profile : Profile
            Halo weak lensing radial profile object
        halo_concentration : float
            Halo concentration
        """
        halo_abundance = cluster_statitstics_modeling.halo_abundance
        Delta_abundance = halo_abundance.overdensity_type
        Delta_profile = profile.core.overdensity_type
        if isinstance(halo_abundance, CastroHaloAbundance) and Delta_profile != "vir":
            raise ValueError(
                f"If the Castro HMF is used, only virial overdensities can be considered. "
                f"The current overdensity in the profile modeling is {Delta_profile}."
            )
        if Delta_abundance != Delta_profile:
            raise ValueError(
                f"The overdensity definition of the mass profile ({Delta_profile}) differs "
                f"from the one adopted for halo abundance modeling ({Delta_abundance}).)"
            )

        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

    def _get_profile(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        effective_inverse_critical_surface_mass_density=None,
    ):
        """Compute weak lensing profile, it can be excess surface density or reduced shear.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.
        effective_inverse_critical_surface_mass_density : numpy.array, None
            The effective inverse of the critical surface density.
            If provided, it must be shape (ztrue, M, radius) and this function
            returns the reduced shear, else it returns the excess surface density.

        Returns
        -------
        wl_profile_mean_values : numpy.ndarray
            Weak lensing quantity (excess surface density or reduced shear) in redshift,
            richness, and radial bins.
        """
        ############################################
        # Get cluster statistics modeling quantities
        ############################################
        # P(lambda_obs_bin,z_obs_bin|M, z): (z_obs_bin,lambda_obs_bin, ztrue, mass)
        window_redshift_lambda_obs = (
            self.cluster_statitstics_modeling.window_redshift_richness_observed(
                z_obs_edges, lambda_obs_edges
            )
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs_bin,z_obs_bin|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_redshift_lambda_obs
            ),
        )

        ########################
        # Compute binned profile
        ########################

        # mass/richness part

        # excess surface mass density : (ztrue, M, radius)
        excess_surface_mass_density = self.profile.excess_surface_mass_density(
            radius_edges[:-1],
            self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
            self.cluster_statitstics_modeling.tabulated_integrands["M"],
            self.halo_concentration,
        )
        # excess surface mass density in a richness bin, integrated on mass w HMF
        # Dimension: (z_obs_bin, lambda_obs_bin, ztrue, radius)
        excess_surface_density_in_window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                excess_surface_mass_density,
                window_redshift_lambda_obs[:, :, :, :, np.newaxis],
            )
        )

        # redshift part

        # Apply effective inverse critical surface mass density if provided
        if effective_inverse_critical_surface_mass_density is not None:
            excess_surface_density_in_window_lambda_obs_mass_integrated *= (
                effective_inverse_critical_surface_mass_density[
                    :, np.newaxis, :, np.newaxis
                ]
            )

        # output : (z_obs, lambda_obs, radius)
        wl_profile_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                excess_surface_density_in_window_lambda_obs_mass_integrated,
            )
        ) / cluster_counts[:, :, np.newaxis]
        return wl_profile_mean_values

    def get_DeltaSigma(self, z_obs_edges, lambda_obs_edges, radius_edges):
        """Compute excess surface density profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.

        Returns
        -------
        deltasigma_mean_values : numpy.ndarray
            Excess surface density in redshift, richness, and radial bins.
        """
        # output : (z_obs, lambda_obs, radius)
        return self._get_profile(
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            effective_inverse_critical_surface_mass_density=None,
        )

    def get_gt(self, z_obs_edges, lambda_obs_edges, radius_edges):
        """Compute reduced shear profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.

        Returns
        -------
        gt_mean_values : numpy.ndarray
           Reduced shear in redshift, richness, and radial bins.
        """
        z_obs_edges_size = len(z_obs_edges) - 1

        # mean z_obs per bin, used to select the tomographic bin index
        z_obs_mean = 0.5 * (z_obs_edges[:-1] + z_obs_edges[1:])

        # Effective inverse critical surface mass density : (z_obs, ztrue)
        effective_inverse_critical_surface_mass_density = np.zeros(
            (
                z_obs_edges_size,
                self.cluster_statitstics_modeling.tabulated_integrands["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_edges_size):
            idx = self.profile.core._get_tomo_bin_index(z_obs_mean[ind_z])
            effective_inverse_critical_surface_mass_density[ind_z] = (
                self.profile.core.sigma_crit_inv_eff(
                    self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
                    idx,
                )
            )

        # output : (z_obs, lambda_obs, radius)
        return self._get_profile(
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            effective_inverse_critical_surface_mass_density,
        )


class ClusterWeakLensingCovariance:
    def __init__(
        self,
        cluster_weak_lensing: ClusterWeakLensing,
        n_eff=15.0,
        sigma_e: float = 0.3,
        nonlinear_perturbations: Perturbations = None,
        source_delta_z: float = 0.05,
        ell: np.ndarray = np.geomspace(1.0, 2.0e5, 1500),
        z_los: np.ndarray = None,
        z_sources_size: int = 100,
        k_nonlinear: np.ndarray = np.geomspace(1.0e-4, 1.0e2, 400),
    ):
        r"""
        Initializes the Gaussian covariance of the cluster tangential shear profile,
        following Wu et al. 2019 (https://arxiv.org/abs/1907.06611), Eq. 10:

        ..math:
            {\rm Cov}[\gamma_t^i(\theta_a), \gamma_t^j(\theta_b)] =
            \frac{1}{4\pi f_{\rm sky}}\int\frac{\ell d\ell}{2\pi}
            \hat J_2(\ell\theta_a)\hat J_2(\ell\theta_b)
            \left[\left(C_\ell^{h_ih_j}+\frac{\delta_{ij}}{n_h^i}\right)
            \left(C_\ell^{\kappa\kappa}+\frac{\sigma_e^2}{n_s}\right)
            + C_\ell^{h_i\kappa}C_\ell^{h_j\kappa}\right]

        where i, j are observed richness bins in the same observed redshift bin.
        The lens sample (redshift distribution, surface density, bias and 1-halo
        term) is predicted from the halo mass function and the selection function
        of the cluster statistics modeling, and the source sample from the profile
        core.

        Parameters
        ----------
        cluster_weak_lensing : ClusterWeakLensing
            Cluster weak lensing object. Its statistics modeling (tables and
            selection function), profile core (overdensity, source redshift
            distribution) and halo concentration are used for the covariance.
        n_eff : float or numpy.ndarray
            Effective source number density (gal/arcmin^2) of the full source
            redshift distribution. Either a float or an array with one value
            per tomographic source bin of the profile core.
        sigma_e : float
            Shape noise per shear component.
        nonlinear_perturbations : Perturbations, None
            Perturbations object providing the non-linear matter power spectrum,
            used for C_ell^kappakappa. If None, the linear matter power spectrum
            of the matter statistics perturbations is used.
        source_delta_z : float
            Redshift buffer of the source selection, z_s > z_obs + source_delta_z.
        ell : numpy.ndarray
            Multipoles used in the ell integration (log-spaced recommended).
        z_los : numpy.ndarray, None
            Line-of-sight redshifts for the C_ell^kappakappa integral. If None,
            50 points linearly spaced in [1e-3, zs_max] are used.
        z_sources_size : int
            Number of source redshift points for the lensing window function.
        k_nonlinear : numpy.ndarray
            Wavenumbers (h/Mpc) where the power spectrum used in C_ell^kappakappa
            is tabulated.

        Notes
        -----
        - The covariance is computed for gamma_t, and used for the reduced shear
          g_t under the weak shear approximation.
        - Radii are physical Mpc/h, converted to angles with the angular diameter
          distance at the mean true redshift of each (z_obs, lambda_obs) bin.
        - The 1-halo term of the halo-matter power spectrum uses a truncated NFW
          profile with the halo concentration of ``cluster_weak_lensing``.
        - Distances assume a flat universe, as in the volume element.
        """
        self.cluster_weak_lensing = cluster_weak_lensing
        self.n_eff = n_eff
        self.sigma_e = sigma_e
        self.nonlinear_perturbations = nonlinear_perturbations
        self.source_delta_z = source_delta_z
        self.ell = ell
        self.z_los = (
            np.linspace(1.0e-3, self.profile_core.zs_max, 50)
            if z_los is None
            else z_los
        )
        self.z_sources_size = z_sources_size
        self.k_nonlinear = k_nonlinear

        # survey solid angle in steradians
        self.survey_solid_angle = (
            self.cluster_statitstics_modeling.area * (np.pi / 180.0) ** 2.0
        )
        # comoving mean densities (units: h^2 Msun / Mpc^3), total matter for the
        # lensing kernel and cdm + baryons (no neutrinos) for the halo model,
        # consistently with the halo mass function
        _rho_crit_0 = (
            derived_cosmology.rho_crit(self.background, 0.0) / self.background.h**2.0
        )
        self.mean_matter_density = (
            self.cluster_statitstics_modeling.matter_statistics.Omega_m_0 * _rho_crit_0
        )
        self.mean_cb_density = (
            self.cluster_statitstics_modeling.matter_statistics.Omega_cb_0 * _rho_crit_0
        )

    @property
    def cluster_statitstics_modeling(self):
        return self.cluster_weak_lensing.cluster_statitstics_modeling

    @property
    def profile_core(self):
        return self.cluster_weak_lensing.profile.core

    @property
    def background(self):
        return self.cluster_statitstics_modeling.matter_statistics.background

    # -------------------
    # auxiliary functions
    # -------------------

    def _comoving_distance(self, z):
        """Comoving distance (units: Mpc/h)."""
        return self.background.comoving_distance(z) * self.background.h

    def _inverse_sigma_crit_comoving(self, z_lens, z_sources):
        r"""Inverse of the comoving critical surface mass density,
        :math:`(1+z_l)^2/\Sigma_{\rm crit}`, set to zero for z_sources <= z_lens.

        Parameters
        ----------
        z_lens : numpy.ndarray
            Lens redshifts, shape (z_lens,).
        z_sources : numpy.ndarray
            Source redshifts, shape (z_sources,).

        Returns
        -------
        numpy.ndarray
            Inverse comoving critical surface mass density (units: Mpc^2 / h / Msun).
            Shape: (z_lens, z_sources).
        """
        fact = (units.SPEED_OF_LIGHT / 1.0e3 / units.MPC_TO_KM) ** 2.0 / (
            4.0 * np.pi * units.GRAVITATIONAL_CONSTANT
        )  # Msun/Mpc
        chi_l = self._comoving_distance(z_lens)[:, np.newaxis]
        chi_s = self._comoving_distance(z_sources)[np.newaxis, :]
        inv_sig_crit = (
            chi_l * (chi_s - chi_l) / chi_s * (1.0 + z_lens[:, np.newaxis]) / fact
        )
        return np.where(
            z_sources[np.newaxis, :] > z_lens[:, np.newaxis], inv_sig_crit, 0.0
        )

    def _source_window(self, z_lens, z_obs_mean):
        r"""Lensing window function of the source sample of each observed
        redshift bin (Wu et al. 2019, Eq. 7):

        ..math:
            F_\kappa(z) = \bar\rho_m \int dz_s\, p(z_s)\, \Sigma_{\rm crit, com}^{-1}(z, z_s)

        with p(z_s) the source redshift distribution of the tomographic bin
        associated to z_obs_mean, restricted to z_s > z_obs_mean + source_delta_z.

        Parameters
        ----------
        z_lens : numpy.ndarray
            Redshifts where the window is evaluated.
        z_obs_mean : numpy.ndarray
            Mean redshift of the observed redshift bins.

        Returns
        -------
        F_kappa : numpy.ndarray
            Lensing window function (units: h / Mpc). Shape: (z_obs, z_lens).
        n_sources : numpy.ndarray
            Source surface number density (units: sr^-1). Shape: (z_obs,).
        """
        F_kappa = np.zeros((z_obs_mean.size, z_lens.size))
        n_sources = np.zeros(z_obs_mean.size)
        n_eff = np.broadcast_to(self.n_eff, self.profile_core.mean_nz.shape)
        for ind_z, z_obs in enumerate(z_obs_mean):
            idx = self.profile_core._get_tomo_bin_index(z_obs)
            z_s = np.linspace(
                z_obs + self.source_delta_z,
                self.profile_core.zs_max,
                self.z_sources_size,
            )
            norm = self.profile_core.n_zs_norM(z_obs, idx, self.source_delta_z)
            p_zs = norm * self.profile_core.n_zs(z_s, idx)
            F_kappa[ind_z] = self.mean_matter_density * simps(
                p_zs * self._inverse_sigma_crit_comoving(z_lens, z_s), x=z_s, axis=1
            )
            # fraction of sources above the cut, gal/arcmin^2 -> gal/sr
            n_sources[ind_z] = n_eff[idx] / norm * (180.0 * 60.0 / np.pi) ** 2.0
        return F_kappa, n_sources

    @staticmethod
    def _limber_power_spectrum(pk_table, k_table, z_chi, ell):
        r"""Evaluate a tabulated power spectrum at the Limber wavenumber
        :math:`k = (\ell + 1/2)/\chi(z)`, log-interpolating in k and setting it to
        zero outside the tabulated k range.

        Parameters
        ----------
        pk_table : numpy.ndarray
            Power spectrum tabulated at (z, k_table) (units: (Mpc/h)^3).
        k_table : numpy.ndarray
            Wavenumbers of the table (units: h/Mpc).
        z_chi : numpy.ndarray
            Comoving distance at each redshift of the table (units: Mpc/h).
        ell : numpy.ndarray
            Multipoles.

        Returns
        -------
        numpy.ndarray
            Power spectrum at the Limber wavenumber. Shape: (z, ell).
        """
        log_k = np.log((ell[np.newaxis, :] + 0.5) / z_chi[:, np.newaxis])
        log_k_table = np.log(k_table)
        return np.array(
            [
                np.exp(np.interp(_log_k, log_k_table, np.log(_pk)))
                * (_log_k >= log_k_table[0])
                * (_log_k <= log_k_table[-1])
                for _log_k, _pk in zip(log_k, pk_table)
            ]
        )

    @staticmethod
    def _nfw_fourier_transform(k, rs, c):
        r"""Normalized Fourier transform of the truncated NFW profile, u(k|M) -> 1
        for k -> 0 (Cooray & Sheth 2002, Eq. 81).

        Parameters
        ----------
        k : numpy.ndarray
            Wavenumbers (units: h/Mpc).
        rs : numpy.ndarray
            Comoving scale radius (units: Mpc/h), broadcastable with k.
        c : float
            Concentration.

        Returns
        -------
        numpy.ndarray
            u(k|M), dimensionless.
        """
        x = k * rs
        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.0 + c) * x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)
        return (
            np.sin(x) * (si_cx - si_x)
            - np.sin(c * x) / ((1.0 + c) * x)
            + np.cos(x) * (ci_cx - ci_x)
        ) / m_nfw

    @staticmethod
    def _bin_averaged_J2(ell, theta_edges):
        r"""Bessel function J_2 averaged in annuli (Wu et al. 2019, Eq. B4):

        ..math:
            \hat J_2 = \frac{2}{\ell^2(\theta_{\max}^2-\theta_{\min}^2)}
            \left[2\left(J_0(\ell\theta_{\min})-J_0(\ell\theta_{\max})\right)
            + \ell\left(\theta_{\min}J_1(\ell\theta_{\min})
            - \theta_{\max}J_1(\ell\theta_{\max})\right)\right]

        Parameters
        ----------
        ell : numpy.ndarray
            Multipoles.
        theta_edges : numpy.ndarray
            Edges of angular bins (radians), shape (..., theta_edges).

        Returns
        -------
        numpy.ndarray
            Bin averaged J_2. Shape: (..., theta, ell).
        """
        theta_min = theta_edges[..., :-1, np.newaxis]
        theta_max = theta_edges[..., 1:, np.newaxis]
        return (
            2.0
            * (
                2.0 * (j0(ell * theta_min) - j0(ell * theta_max))
                + ell
                * (theta_min * j1(ell * theta_min) - theta_max * j1(ell * theta_max))
            )
            / (ell**2.0 * (theta_max**2.0 - theta_min**2.0))
        )

    # ------------------------
    # angular power spectra
    # ------------------------

    def _lens_sample(self, z_obs_edges, lambda_obs_edges):
        r"""Lens sample quantities predicted from the halo mass function.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins.

        Returns
        -------
        dict
            Contains :

                * window_redshift_lambda_obs (numpy.ndarray) : P(lambda_obs_bin, z_obs_bin|M, ztrue), (z_obs, lambda_obs, ztrue, M).
                * cluster_counts (numpy.ndarray) : Number counts N, (z_obs, lambda_obs).
                * dN_dz (numpy.ndarray) : Normalized true redshift distribution p(ztrue) = dN/dz / N, (z_obs, lambda_obs, ztrue).
                * bias_dN_dz (numpy.ndarray) : b_eff(ztrue) * p(ztrue), (z_obs, lambda_obs, ztrue).
                * z_eff (numpy.ndarray) : Mean true redshift, (z_obs, lambda_obs).
        """
        csm = self.cluster_statitstics_modeling
        dv_dz = csm.tabulated_integrands["dv/dz(ztrue)"]

        # P(lambda_obs_bin,z_obs_bin|M, z) : (z_obs, lambda_obs, ztrue, M)
        window_redshift_lambda_obs = csm.window_redshift_richness_observed(
            z_obs_edges, lambda_obs_edges
        )
        # comoving number density of the lens sample : (z_obs, lambda_obs, ztrue)
        number_density = csm.integrate_probe_function_in_mass(
            np.ones((1, 1)), window_redshift_lambda_obs
        )
        # bias weighted comoving number density : (z_obs, lambda_obs, ztrue)
        bias_number_density = csm.integrate_probe_function_in_mass(
            csm.tabulated_integrands["bias(ztrue,M)"], window_redshift_lambda_obs
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = csm.integrate_probe_function_in_redshift(number_density)

        _norm = np.divide(
            dv_dz,
            cluster_counts[:, :, np.newaxis],
            out=np.zeros(number_density.shape),
            where=cluster_counts[:, :, np.newaxis] != 0,
        )
        dN_dz = number_density * _norm
        return {
            "window_redshift_lambda_obs": window_redshift_lambda_obs,
            "cluster_counts": cluster_counts,
            "dN_dz": dN_dz,
            "bias_dN_dz": bias_number_density * _norm,
            "z_eff": simps(
                dN_dz * csm.tabulated_integrands["ztrue"],
                x=csm.tabulated_integrands["ztrue"],
                axis=2,
            ),
        }

    def _cl_hh(self, lens_sample, pk_lin_limber):
        r"""Halo-halo angular power spectrum (Wu et al. 2019, Eq. 5), with
        :math:`P_{hh} = b_{\rm eff}^i b_{\rm eff}^j P_{\rm lin}`:

        ..math:
            C_\ell^{h_ih_j} = \int dz \frac{p_i(z)b_i(z)\,p_j(z)b_j(z)}{dV/dzd\Omega}
            P_{\rm lin}\left(\frac{\ell+1/2}{\chi(z)}, z\right)

        Returns
        -------
        numpy.ndarray
            Shape: (z_obs, lambda_obs, lambda_obs, ell).
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        dV_dzdO = derived_cosmology.dV_dzdO(self.background, z_true, hubble_units=True)
        bias_dN_dz = lens_sample["bias_dN_dz"]
        return simps(
            (
                bias_dN_dz[:, :, np.newaxis, :]
                * bias_dN_dz[:, np.newaxis, :, :]
                / dV_dzdO
            )[..., np.newaxis]
            * pk_lin_limber,
            x=z_true,
            axis=3,
        )

    def _cl_hkappa(self, lens_sample, pk_lin_limber, F_kappa_ztrue):
        r"""Halo-convergence angular power spectrum (Wu et al. 2019, Eq. 8), with
        the halo-matter power spectrum given by the 1-halo (NFW) and 2-halo terms:

        ..math:
            C_\ell^{h_i\kappa} = \int dz\, p_i(z) \frac{F_\kappa(z)}{\chi^2(z)}
            \left[\frac{1}{n_i(z)}\int dM \frac{dn}{dM} W_i(M,z)\frac{M}{\bar\rho_{cb}}u(k|M,z)
            + b_i(z) P_{\rm lin}(k, z)\right]_{k=(\ell+1/2)/\chi(z)}

        Returns
        -------
        numpy.ndarray
            Shape: (z_obs, lambda_obs, ell).
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        mass = csm.tabulated_integrands["M"]
        chi = self._comoving_distance(z_true)
        window = lens_sample["window_redshift_lambda_obs"]

        # comoving NFW scale radius : (ztrue, M, 1)
        _, RDelta, _ = self.profile_core.surface_mass_density_args(
            np.zeros(1), z_true, mass
        )
        rs = (
            RDelta
            * (1.0 + z_true[:, np.newaxis, np.newaxis])
            / self.cluster_weak_lensing.halo_concentration
        )
        # M/rho_cb u(k|M) : (ztrue, M, ell)
        k_limber = (self.ell[np.newaxis, :] + 0.5) / chi[:, np.newaxis]
        mass_profile = (
            mass[np.newaxis, :, np.newaxis]
            / self.mean_cb_density
            * self._nfw_fourier_transform(
                k_limber[:, np.newaxis, :],
                rs,
                self.cluster_weak_lensing.halo_concentration,
            )
        )

        # 1-halo term times p_i(z) : (z_obs, lambda_obs, ztrue, ell)
        _norm = np.divide(
            csm.tabulated_integrands["dv/dz(ztrue)"],
            lens_sample["cluster_counts"][:, :, np.newaxis],
            out=np.zeros(lens_sample["dN_dz"].shape),
            where=lens_sample["cluster_counts"][:, :, np.newaxis] != 0,
        )
        p_hm_1h = (
            np.concatenate(
                [
                    csm.integrate_probe_function_in_mass(
                        mass_profile, window[ind_z : ind_z + 1, :, :, :, np.newaxis]
                    )
                    for ind_z in range(window.shape[0])
                ]
            )
            * _norm[..., np.newaxis]
        )
        # 2-halo term times p_i(z) : (z_obs, lambda_obs, ztrue, ell)
        p_hm_2h = lens_sample["bias_dN_dz"][..., np.newaxis] * pk_lin_limber

        return simps(
            (F_kappa_ztrue / chi**2.0)[:, np.newaxis, :, np.newaxis]
            * (p_hm_1h + p_hm_2h),
            x=z_true,
            axis=2,
        )

    def _cl_kappakappa(self, F_kappa_los):
        r"""Convergence angular power spectrum (Wu et al. 2019, Eq. 6):

        ..math:
            C_\ell^{\kappa\kappa} = \int dz \frac{d\chi}{dz}\frac{F_\kappa^2(z)}{\chi^2(z)}
            P_{\rm NL}\left(\frac{\ell+1/2}{\chi(z)}, z\right)

        Returns
        -------
        numpy.ndarray
            Shape: (z_obs, ell).
        """
        h = self.background.h
        chi = self._comoving_distance(self.z_los)
        dchi_dz = (
            units.SPEED_OF_LIGHT
            / 1.0e3
            / self.background.hubble_parameter(self.z_los)
            * h
        )

        perturbations = (
            self.cluster_statitstics_modeling.matter_statistics.perturbations
            if self.nonlinear_perturbations is None
            else self.nonlinear_perturbations
        )
        pk_table = perturbations.matter_power_spectrum(
            self.z_los, self.k_nonlinear, hubble_units=True, k_hunit=True
        )
        pk_limber = self._limber_power_spectrum(
            pk_table, self.k_nonlinear, chi, self.ell
        )

        return simps(
            (dchi_dz * F_kappa_los**2.0 / chi**2.0)[:, :, np.newaxis] * pk_limber,
            x=self.z_los,
            axis=1,
        )

    # ----------------------
    # covariance
    # ----------------------

    def get_gt_covariance(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        cross_richness=True,
        return_terms=False,
    ):
        """Compute the Gaussian covariance of the reduced shear profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile (units: physical Mpc/h).
        cross_richness : bool
            If True, computes also the covariance between different richness
            bins in the same redshift bin, else only the auto richness blocks.
        return_terms : bool
            If True, returns also the shape noise, LSS and intrinsic terms
            (Wu et al. 2019, Eqs. 11-13), which sum to the covariance.

        Returns
        -------
        cov_gt : numpy.ndarray
            Covariance of the reduced shear in redshift, richness and radial bins,
            diagonal in redshift bins.
            Dimension: (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius).
        terms (optional) : dict
            Dictionary with the "shape", "lss" and "intr" terms, same dimension
            as cov_gt. Returned only when `return_terms` is true.
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        z_obs_edges_size = len(z_obs_edges) - 1
        lambda_obs_edges_size = len(lambda_obs_edges) - 1
        radius_edges_size = len(radius_edges) - 1
        z_obs_mean = 0.5 * (z_obs_edges[:-1] + z_obs_edges[1:])

        ##############################
        # Lens and source sample
        ##############################
        lens_sample = self._lens_sample(z_obs_edges, lambda_obs_edges)
        # lens surface density in sr^-1 : (z_obs, lambda_obs)
        n_lens = lens_sample["cluster_counts"] / self.survey_solid_angle

        # lensing window functions : (z_obs, ztrue), (z_obs, z_los)
        F_kappa_ztrue, n_sources = self._source_window(z_true, z_obs_mean)
        F_kappa_los, _ = self._source_window(self.z_los, z_obs_mean)

        ##############################
        # Angular power spectra
        ##############################
        chi_true = self._comoving_distance(z_true)
        k_table = csm.tabulated_integrands["k"]
        pk_lin_limber = self._limber_power_spectrum(
            csm.matter_statistics.matter_power_spectrum_cb(z_true, k_table),
            k_table,
            chi_true,
            self.ell,
        )
        # (z_obs, lambda_obs, lambda_obs, ell)
        cl_hh = self._cl_hh(lens_sample, pk_lin_limber)
        # (z_obs, lambda_obs, ell)
        cl_hk = self._cl_hkappa(lens_sample, pk_lin_limber, F_kappa_ztrue)
        # (z_obs, ell)
        cl_kk = self._cl_kappakappa(F_kappa_los)

        if not cross_richness:
            cl_hh *= np.identity(lambda_obs_edges_size)[np.newaxis, :, :, np.newaxis]

        # lens shot noise : (z_obs, lambda_obs, lambda_obs)
        shot_noise_lens = np.divide(
            np.identity(lambda_obs_edges_size)[np.newaxis, :, :],
            n_lens[:, :, np.newaxis],
            out=np.zeros(
                (z_obs_edges_size, lambda_obs_edges_size, lambda_obs_edges_size)
            ),
            where=n_lens[:, :, np.newaxis] != 0,
        )
        shape_noise_sources = (self.sigma_e**2.0 / n_sources)[
            :, np.newaxis, np.newaxis, np.newaxis
        ]

        # integrands in ell : (z_obs, lambda_obs, lambda_obs, ell)
        # the lens shot noise x shape noise term is computed analytically below
        _integrands = {
            "shape": cl_hh * shape_noise_sources,
            "lss": (cl_hh + shot_noise_lens[..., np.newaxis])
            * cl_kk[:, np.newaxis, np.newaxis, :],
            "intr": cl_hk[:, :, np.newaxis, :] * cl_hk[:, np.newaxis, :, :],
        }
        if not cross_richness:
            _integrands["intr"] *= np.identity(lambda_obs_edges_size)[
                np.newaxis, :, :, np.newaxis
            ]

        ##############################
        # Hankel transform in radial bins
        ##############################
        # radius to angle at the mean true redshift : (z_obs, lambda_obs, radius_edges)
        D_A = (
            self.background.angular_diameter_distance(lens_sample["z_eff"].flatten())
            * self.background.h
        ).reshape(lens_sample["z_eff"].shape)
        theta_edges = radius_edges[np.newaxis, np.newaxis, :] / D_A[:, :, np.newaxis]
        # (z_obs, lambda_obs, radius, ell)
        hat_J2 = self._bin_averaged_J2(self.ell, theta_edges)

        # ell d ell / 2pi -> ell^2 / 2pi dln(ell), trapezoidal weights in ln(ell)
        _dlnell = np.diff(np.log(self.ell))
        _weights = np.zeros(self.ell.size)
        _weights[:-1] += 0.5 * _dlnell
        _weights[1:] += 0.5 * _dlnell
        _weights *= self.ell**2.0 / (2.0 * np.pi)

        fsky = self.survey_solid_angle / (4.0 * np.pi)
        terms = {
            name: np.einsum(
                "ziaL,zjbL,zijL,L->zijab", hat_J2, hat_J2, integrand, _weights
            )
            / (4.0 * np.pi * fsky)
            for name, integrand in _integrands.items()
        }

        # lens shot noise x shape noise, using int ell dell/2pi J2_a J2_b = delta_ab/A_a
        # (Wu et al. 2019, App. B): sigma_e^2 / (N_h n_s A_a)
        annulus_area = np.pi * (
            theta_edges[..., 1:] ** 2.0 - theta_edges[..., :-1] ** 2.0
        )
        terms["shape"] += (
            shot_noise_lens[..., np.newaxis, np.newaxis]
            * shape_noise_sources[..., np.newaxis]
            / (4.0 * np.pi * fsky)
            * (np.identity(radius_edges_size) / annulus_area[:, :, np.newaxis, :])[
                :, np.newaxis, :, :, :
            ]
        )

        # make it (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius), diagonal in z_obs
        _z_identity = np.identity(z_obs_edges_size)[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]
        terms = {
            name: _z_identity * term[np.newaxis, ...] for name, term in terms.items()
        }
        cov_gt = terms["shape"] + terms["lss"] + terms["intr"]

        if return_terms:
            return cov_gt, terms
        return cov_gt

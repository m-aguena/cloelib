# General imports
# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson
import scipy.special as spc

from cloelib.observables.clusters.halo_mass_observable import (
    HaloMassObservable,
)


class AnalyticSelectionFunction:
    def __init__(
        self,
        halo_mass_observable: HaloMassObservable,
        sig_lambda_norm: float,
        sig_lambda_z: float,
        sig_lambda_exponent: float,
        mu_lambda_norm: float,
        mu_lambda_z: float,
        mu_lambda_0: float,
        tau_lambda_norm: float,
        tau_lambda_z: float,
        tau_lambda_exponent: float,
        fprj_lambda_norm: float,
        fprj_lambda_z: float,
        fprj_lambda_exponent: float,
        sig_z_exponent: float,
        sig_z_lambda_norm: float,
        sig_z_z_norm: float,
        z_tab_integ: int,
        lambda_tab_integ: list,
    ):
        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        Parameters
        ----------
        halo_mass_observable: HaloMassObservable,
            Object that contains the distribution of true richness given mass
        sig_lambda_norm: float
            Amplitude of the observed proxy - true proxy relation
        sig_lambda_z: float
            Redshift evolution of the observed proxy - true proxy relation
        sig_lambda_exponent: float
            Exponential evolution of the observed proxy - true proxy relation
        sig_z_z: float
            Amplitude of the observed redshift - true redshift relation
        sig_z_lambda: float
            Proxy evolution of the observed redshift - true redshift relation
        z_tab_integ : int
            Number of points to be used for z_obs integration.
        lambda_tab_integ : List
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_edges.
        """
        self.halo_mass_observable = halo_mass_observable
        self.sig_lambda_norm = sig_lambda_norm
        self.sig_lambda_z = sig_lambda_z
        self.sig_lambda_exponent = sig_lambda_exponent
        self.mu_lambda_norm = mu_lambda_norm
        self.mu_lambda_z = mu_lambda_z
        self.mu_lambda_0 = mu_lambda_0
        self.tau_lambda_norm = tau_lambda_norm
        self.tau_lambda_z = tau_lambda_z
        self.tau_lambda_exponent = tau_lambda_exponent
        self.fprj_lambda_norm = fprj_lambda_norm
        self.fprj_lambda_z = fprj_lambda_z
        self.fprj_lambda_exponent = fprj_lambda_exponent
        self.sig_z_exponent = sig_z_exponent
        self.sig_z_lambda_norm = sig_z_lambda_norm
        self.sig_z_z_norm = sig_z_z_norm
        self.z_tab_integ = z_tab_integ
        self.lambda_tab_integ = lambda_tab_integ

    def _scatter_lambda_obs(self, z, lambda_true):
        r"""
        Statistical uncertainty on the observed mass proxy.

        Computes the scatter of the observed richness PDF
        at the requested true redshift and richness points.

        ..math:
            \sigma_{\lambda_{\rm obs}}(\lambda_{\rm true},z_{\rm true}) =
            (\sigma_0 +z_{\rm true}*\sigma_z) * \lambda_{\rm true}^{\sigma_{\rm exp} }

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        scatter_lbobs_lbdz: numpy.ndarray
            Statistical uncertainty on the observed mass proxy.
        """
        return (
            self.sig_lambda_norm + self.sig_lambda_z * z
        ) * lambda_true**self.sig_lambda_exponent

    def _mu_lambda_obs(self, z, lambda_true, lambda_pivot=25.0):
        r"""
        mean observed richness at the requested true redshift and richness points.

        ..math:
            \mu_{\lambda_{\rm obs}}(\lambda_{\rm true},z_{\rm true}) = max \left[ 1,
            \mu_{\lambda_0} + (\mu_0 +z_{\rm true}*\mu_z) * (\lambda_{\rm true} -lambda_pivot) \right]


        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        scatter_lbobs_lbdz: numpy.ndarray
            Mean observed mass proxy.
        """
        mu = (self.mu_lambda_norm + self.mu_lambda_z * z) * (
            lambda_true - lambda_pivot
        ) + self.mu_lambda_0
        return np.clip(mu, 1.0, None)

    def _tau_lambda_obs(self, z, lambda_true):
        r"""
        Exponential slope of the observed richness distribution at the
        requested true richness points.

        ..math:
            \tau_{\lambda_{\rm obs}}(\lambda_{\rm true},z_{\rm true}) =
            \frac{\tau_0 + \tau_z z_{\rm true}}{\lambda_{\rm true}^{\tau_{\rm exp}}}

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        tau_lbobs: numpy.ndarray
            Exponential slope of the observed richness distribution.
        """
        return (
            self.tau_lambda_norm + self.tau_lambda_z * z
        ) / lambda_true**self.tau_lambda_exponent

    def _fprj_lambda_obs(self, z, lambda_true):
        r"""
        Projected cluster fraction at the requested true richness points.

        ..math:
            f_{\rm prj}(\lambda_{\rm true}) =
            f_{\rm prj}_0 * \lambda_{\rm true}^{[f_{\rm prj}_{\rm exp} + + {f_{\rm prj}_z z_{\rm true}}

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        fprj_lbobs: numpy.ndarray
            Projected cluster fraction.
        """
        return self.fprj_lambda_norm * lambda_true ** (
            self.fprj_lambda_exponent + self.fprj_lambda_z * z
        )

    @staticmethod
    def _gaussian(value, mean, sigma):
        return np.exp(-((value - mean) ** 2.0) / (2.0 * sigma**2.0)) / (
            np.sqrt(2.0 * np.pi * sigma**2.0)
        )

    def _prob_lambda_obs(self, z, lambda_true, lambda_obs):
        r"""
        Observed mass proxy PDF.

        Computes the observed richness PDF at the requested
        true richness, true redshift, and observed richness points

        ..math:
        P(\lambda_{\rm obs} | \lambda_{\rm true},z_{\rm true})= (1-f^{\rm prj}) \mathcal{N}~(\mu,\sigma) +
        + f^{\rm prj} \frac{\tau}{2} \exp\left[ \frac{\tau}{2} (2\mu + \tau\sigma^2 - 2 \lambda_{\rm obs})\right]
        {\rm erfc} \left( \frac{\mu +\tau\sigma^2-\lambda_{\rm obs}}{\sqrt{2}\sigma}\right)

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.
        lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        prob_lambda_obs: numpy.ndarray
            Observed mass proxy PDF.
        """

        mu = self._mu_lambda_obs(z, lambda_true)
        sig_pure = self._scatter_lambda_obs(z, lambda_true)
        sig2_l = sig_pure**2.0
        tau = self._tau_lambda_obs(z, lambda_true)
        f_prj = self._fprj_lambda_obs(z, lambda_true)
        erfc_arg1 = (mu + tau * sig2_l - lambda_obs) / np.sqrt(2 * sig2_l)
        exptau = np.exp(
            0.5 * tau * (2.0 * mu + tau * sig2_l - 2.0 * lambda_obs), dtype="float128"
        )
        pdf = (1.0 - f_prj) * self._gaussian(
            lambda_obs, mu, sig_pure
        ) + exptau * spc.erfc(erfc_arg1) * (f_prj * tau / 2.0)
        return pdf

    def scatter_z_obs(self, lambda_obs, z, lambda_true=None, z_pivot=0.2):
        r"""
        Statistical uncertainty on the observed redshift.

        Computes the scatter of the observed redshift PDF
        at the requested true redshift, true richness, and observed richness points.

        ..math:
            \sigma_{z_{\rm obs}}(\lambda_{\rm obs}, \lambda_{\rm true}, z_{\rm true}) =
            \langle\lambda_{\rm obs}\rangle^{\alpha} \cdot a +
            b \cdot \frac{z_{\rm true}}{z_{\rm pivot}} \cdot
            \frac{\lambda_{\rm obs} - \langle\lambda_{\rm obs}\rangle}
                {\langle\lambda_{\rm obs}\rangle}

        where :math:`\langle\lambda_{\rm obs}\rangle =
        \mu_{\lambda_{\rm obs}}(\lambda_{\rm true}, z_{\rm true})`.

        Parameters
        ----------
        lambda_obs: numpy.ndarray
            Observed richness points.
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points. If None assume _mu_lambda_obs = lambda_obs
        z_pivot: float
            Pivot redshift.

        Returns
        -------
        scatter_z_obs: numpy.ndarray
            Statistical uncertainty on the observed redshift.
        """
        if lambda_true is None:
            return (
                lambda_obs**self.sig_z_exponent
                * self.sig_z_lambda_norm
                * np.ones(z.shape)
            )
        mu = self._mu_lambda_obs(z, lambda_true)
        sig_base = mu**self.sig_z_exponent * self.sig_z_lambda_norm
        sig_slope = self.sig_z_z_norm * (z / z_pivot) * (lambda_obs - mu) / mu
        return np.where(lambda_obs >= mu, sig_base + sig_slope, sig_base)

    def _prob_z_obs(self, z_obs, lambda_obs, z_true, lambda_true):
        r"""
        Observed redshift PDF.

        Computes the observed redshift PDF at the requested
        true richness, true redshift, and observed redshift points.

        Parameters
        ----------
        z_obs: numpy.ndarray
            Observed redshift points.
        lambda_obs: numpy.ndarray
            Observed richness points.
        z_true: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        numpy.ndarray
            Observed redshift PDF.
        """
        return self._gaussian(
            z_obs, z_true, self.scatter_z_obs(lambda_obs, z_true, lambda_true)
        )

    def window_zob_given_lob_ztr_ltr(
        self, z_obs_edges, lambda_obs, z_true, lambda_true
    ):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z_{\rm obs}}(\lambda_{\rm obs},\lambda_{\rm true}, z_{\rm true}) =
            \int_{\Delta z_{\rm obs}}dz_{\rm obs} P(z_{\rm obs}|\lambda_{\rm obs}, \lambda_{\rm true}, z_{\rm true})
            = \frac{1}{2} \left[
                {\rm erf}\left(\frac{z_{\rm obs}^{\rm high} - z_{\rm true}}
                                {\sqrt{2}\,\sigma_{z_{\rm obs}}}\right) -
                {\rm erf}\left(\frac{z_{\rm obs}^{\rm low} - z_{\rm true}}
                                {\sqrt{2}\,\sigma_{z_{\rm obs}}}\right)
            \right]

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs : numpy.ndarray
            Observed richness to compute the window.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : numpy.ndarray
            True richness to compute the window.

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, z_true) in z_obs bins.
            Dimensions: (z_obs_bins, lambda_obs, z_true, lambda_true).
        """
        # bin edges, shape: (z_obs_bins,) each
        z_obs_low = z_obs_edges[:-1]
        z_obs_high = z_obs_edges[1:]

        # reshape for broadcasting: (z_obs_bins, lambda_obs, z_true, lambda_true)
        _z_obs_low = z_obs_low[:, np.newaxis, np.newaxis, np.newaxis]
        _z_obs_high = z_obs_high[:, np.newaxis, np.newaxis, np.newaxis]
        _lambda_obs = lambda_obs[np.newaxis, :, np.newaxis, np.newaxis]
        _z_true = z_true[np.newaxis, np.newaxis, :, np.newaxis]
        _lambda_true = lambda_true[np.newaxis, np.newaxis, np.newaxis, :]

        sig = self.scatter_z_obs(_lambda_obs, _z_true, _lambda_true)

        arg_high = (_z_obs_high - _z_true) / (np.sqrt(2.0) * sig)
        arg_low = (_z_obs_low - _z_true) / (np.sqrt(2.0) * sig)

        return 0.5 * (spc.erf(arg_high) - spc.erf(arg_low))

    def window_z_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, lambda_true=None
    ):
        r"""Compute the redshift window function of each observed redshift bin,
        marginalized over lambda_true:

        ..math:
            W_{\Delta z_{\rm obs},\lambda_{\rm obs}}(z_{\rm true}) =
            \int_{\Delta z_{\rm obs}}dz_{\rm obs}
            \int_{\Delta \lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_0^{\infty}
            d \lambda_{\rm true} P(\lambda_{\rm obs},z_{\rm obs}| \lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.
            If not provided computed internally to the function

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(Delta_lambda_obs, Delta_z_obs| lambda_true, z_true) over lambda_true.
            Dimensions: (z_obs_bins, lambda_obs, z_true).
        """

        if lambda_true is None:
            lambda_true = np.linspace(5.0, 300.0, 50)

        # Dimensions: (z_obs_bins, lambda_obs_bin, z_true, lambda_true)
        w_Dzob_Dlob__ztr_ltr = self.window_z_lambda_observed(
            z_obs_edges, lambda_obs_edges, z_true, lambda_true
        )
        return simpson(w_Dzob_Dlob__ztr_ltr, x=lambda_true, axis=-1)

    def window_lambda_observed(self, lambda_obs_edges, z_true, lambda_true):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(z_{\rm true}, \lambda_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}} d\lambda_{\rm obs}\,
            P(\lambda_{\rm obs} | \lambda_{\rm true}, z_{\rm true})

        The CDF of the distribution has a closed form:

        ..math:
            F(\lambda_{\rm obs}) =
            (1 - f^{\rm prj}) \frac{1}{2}\left[1 + {\rm erf}\left(
                \frac{\lambda_{\rm obs} - \mu}{\sqrt{2}\sigma}
            \right)\right]
            + f^{\rm prj} \exp\left[\frac{\tau}{2}(2\mu + \tau\sigma^2 - 2\lambda_{\rm obs})\right]
            {\rm erfc}\left(\frac{\mu + \tau\sigma^2 - \lambda_{\rm obs}}{\sqrt{2}\sigma}\right)

        and the bin integral is simply :math:`F(\lambda_{\rm obs}^{\rm high}) - F(\lambda_{\rm obs}^{\rm low})`.

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of observed richness bins.
        z_true : numpy.ndarray
            True redshift points.
        lambda_true : numpy.ndarray
            True richness points.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|lambda_true, z_true) in lambda_obs bins.
            Dimensions: (lambda_obs_bins, z_true, lambda_true).
        """
        # reshape edges for broadcasting: (lambda_obs_bins, z_true, lambda_true)
        _lob_low = lambda_obs_edges[:-1, np.newaxis, np.newaxis]
        _lob_high = lambda_obs_edges[1:, np.newaxis, np.newaxis]
        _z_true = z_true[np.newaxis, :, np.newaxis]
        _lambda_true = lambda_true[np.newaxis, np.newaxis, :]

        mu = self._mu_lambda_obs(_z_true, _lambda_true)
        sig = self._scatter_lambda_obs(_z_true, _lambda_true)
        sig2 = sig**2.0
        tau = self._tau_lambda_obs(_z_true, _lambda_true)
        f_prj = self._fprj_lambda_obs(_z_true, _lambda_true)

        def _cdf(lob):
            # Gaussian CDF: Phi((lob-mu)/sigma)
            gauss_cdf = 0.5 * (1.0 + spc.erf((lob - mu) / (np.sqrt(2.0) * sig)))

            # EMG correction: exp[-tau*(lob-mu) + 0.5*tau^2*sig^2] * Phi((lob-mu)/sig - tau*sig)
            exp_arg = -tau * (lob - mu) + 0.5 * tau**2.0 * sig2
            phi_arg = (lob - mu) / (np.sqrt(2.0) * sig) - tau * sig / np.sqrt(2.0)
            exp_cdf = np.exp(exp_arg, dtype="float128") * 0.5 * (1.0 + spc.erf(phi_arg))

            return (1.0 - f_prj) * gauss_cdf - f_prj * exp_cdf

        return (_cdf(_lob_high) - _cdf(_lob_low)).astype(float)

    def window_richness_observed(
        self,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true,
    ):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            P(\lambda_{\rm obs}|\lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|\lambda_{\rm true}, z_true) in lambda_obs bins.
            Dimensions: (lambda_obs_edges-1, z_true, mass).
        """

        lambda_obs_bins_size = len(lambda_obs_edges) - 1
        if len(self.lambda_tab_integ) != lambda_obs_bins_size:
            raise ValueError(
                f"Number of bins from lambda_obs_edges ({lambda_obs_bins_size})"
                " is different from internal lambda_tab_integ"
                f" ({(len(self.lambda_tab_integ))}) setup!"
            )

        ################################################
        # Compute P(Delta lobs|ltrue, ztrue)
        ################################################
        windows_lambda_obs_lambda_true = self.window_lambda_observed(
            lambda_obs_edges, z_true, lambda_true
        )

        ################################################
        # Compute P(Delta lobs|mass, ztrue)
        ################################################
        pdf_mass_richness_scaling = self.halo_mass_observable.pdf_richness(
            z_true, mass, lambda_true
        )

        # return simpson(
        return np.trapezoid(
            pdf_mass_richness_scaling[np.newaxis, :, :, :]  # (1, z, M, ltr)
            * windows_lambda_obs_lambda_true[:, :, np.newaxis, :],  # (lobs, z, 1, ltr)
            x=lambda_true,
            axis=-1,
        )

    def window_z_lambda_observed(
        self,
        z_obs_edges,
        lambda_obs_edges,
        z_true,
        lambda_true,
    ):
        r"""
        Computes the window function for observed redshift and richness bins
        at fixed true richness and redshift:

        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(\lambda_{\rm true}, z_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}} d\lambda_{\rm obs}
            P(\lambda_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \int_{\Delta z_{\rm obs}} d z_{\rm obs}
            P(z_{\rm obs}|\lambda_{\rm obs}, \lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : numpy.ndarray
            True richness to compute the window.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_bins, lambda_obs_bins, z_true, lambda_true).
        """
        lambda_obs_bins_size = len(lambda_obs_edges) - 1
        z_obs_bins_size = len(z_obs_edges) - 1

        if len(self.lambda_tab_integ) != lambda_obs_bins_size:
            raise ValueError(
                f"Number of bins from lambda_obs_edges ({lambda_obs_bins_size})"
                " is different from internal lambda_tab_integ"
                f" ({len(self.lambda_tab_integ)}) setup!"
            )

        # output: (z_obs_bins, lambda_obs_bins, z_true, lambda_true)
        window = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                z_true.size,
                lambda_true.size,
            )
        )

        for i_lob in range(lambda_obs_bins_size):
            lob_tab = np.linspace(
                lambda_obs_edges[i_lob],
                lambda_obs_edges[i_lob + 1],
                self.lambda_tab_integ[i_lob],
            )  # (n_tab,)

            # # w_zob at each lambda_obs quadrature point
            # # shape: (n_tab, z_obs_bins, z_true, lambda_true)
            w_zob_tab = self.window_zob_given_lob_ztr_ltr(
                z_obs_edges, lob_tab, z_true, lambda_true
            ).transpose(1, 0, 2, 3)

            # P(lob|ltr, ztr) at each quadrature point
            # shape: (n_tab, z_true, lambda_true)
            _lob = lob_tab[:, np.newaxis, np.newaxis]
            _ztr = z_true[np.newaxis, :, np.newaxis]
            _ltr = lambda_true[np.newaxis, np.newaxis, :]
            p_lob = self._prob_lambda_obs(_ztr, _ltr, _lob)

            # integrand: (n_tab, z_obs_bins, z_true, lambda_true)
            integrand = p_lob[:, np.newaxis, :, :] * w_zob_tab

            # integrate over lambda_obs -> (z_obs_bins, z_true, lambda_true)
            window[:, i_lob, :, :] = simpson(integrand, x=lob_tab, axis=0)

        return window

    def window_redshift_richness_observed(
        self,
        z_obs_edges,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true=None,
    ):
        r"""
        Computes the window function for observed redshift and richness bins:

        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.
            If not provided, defined internally in the function

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_bins, lambda_obs_bins, z_true, mass).
        """
        lambda_obs_bins_size = len(lambda_obs_edges) - 1
        z_obs_bins_size = len(z_obs_edges) - 1

        if len(self.lambda_tab_integ) != lambda_obs_bins_size:
            raise ValueError(
                f"Number of bins from lambda_obs_edges ({lambda_obs_bins_size})"
                " is different from internal lambda_tab_integ"
                f" ({len(self.lambda_tab_integ)}) setup!"
            )

        # output array: (z_obs_bins, lambda_obs_bins, z_true, mass)
        window = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                z_true.size,
                mass.size,
            )
        )

        # Define the lambda_true grids for integration
        if lambda_true is None:
            mean_ltr_given_lob_ztr = (
                lambda_obs_edges[[0, -1], np.newaxis] - self.mu_lambda_0
            ) / (self.mu_lambda_norm + self.mu_lambda_z * z_true[np.newaxis, :]) + 25.0
            np.amin(mean_ltr_given_lob_ztr) * 0.5
            hi_ltr = np.amax(mean_ltr_given_lob_ztr) * 1.5
            lambda_true = np.linspace(1.0, hi_ltr, 50)

        # window integrated over lob and zob bin
        # shape: (z_obs_bin, lambda_obs_bin, z_true, lambda_true)
        w_lob_zob = self.window_z_lambda_observed(
            z_obs_edges, lambda_obs_edges, z_true, lambda_true
        )

        # --- marginalize over lambda_true ---
        # P(lambda_true|M, z_true) shape: (z_true, mass, lambda_true)
        p_ltr = p_ltr = self.halo_mass_observable.pdf_richness(
            z_true, mass, lambda_true
        )

        # w_lob_zob: (z_obs_bins, lambda_obs_bins, z_true, lambda_true)
        # p_ltr:     (z_true, mass, lambda_true)
        integrand_ltr = (
            w_lob_zob[
                :, :, :, np.newaxis, :
            ]  # (z_obs_bins, lambda_obs_bin, z_true, 1, lambda_true)
            * p_ltr[
                np.newaxis, np.newaxis, :, :, :
            ]  # (1, 1, z_true, mass, lambda_true)
        )
        # integrate over lambda_true -> (z_obs_bins, z_true, mass)
        window = simpson(integrand_ltr, x=lambda_true, axis=-1)

        return window

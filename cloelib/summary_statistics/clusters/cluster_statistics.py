# cloelib imports
# General imports
# import interpax
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.hmf_bias import HMFBias
from cloelib.observables.clusters.profile import Profile
from cloelib.observables.clusters.selection_function import SelectionFunction

# import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""


class ClusterStatistics:
    def __init__(
        self,
        perturbations: Perturbations,
        halostatistics: HaloStatistics,
        selectionfunction: SelectionFunction,
        hmfbias: HMFBias,
        profile: Profile,
        clustering: HaloClustering,
        covariance: HaloCovariance,
        halo_concentration: float,
        integ_k_arr: np.ndarray,
        integ_mass_arr: np.ndarray,
        integ_lambda_arr: np.ndarray,
        integ_ztrue_arr: np.ndarray,
        area: float = 10313,
    ):
        """
        Initializes the cluster counts

        Parameters:
        - ....

        """
        # observable objects
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.halostatistics = halostatistics
        self.hmfbias = hmfbias
        self.selectionfunction = selectionfunction
        self.profile = profile
        self.clustering = clustering
        self.covariance = covariance

        # internal values
        self.area = area
        self.halo_concentration = halo_concentration

        # integration variables
        self.integ_k_arr = integ_k_arr  # k array
        self.integ_mass_arr = integ_mass_arr  # mass array in Msun h^-1
        self.integ_lambda_arr = integ_lambda_arr  # true richness array
        self.integ_ztrue_arr = integ_ztrue_arr  # true redshift array

        # observational bins
        self.bins = {
            name: Bin(name)
            for name in (
                "z_obs_nc",
                "lambda_obs_nc",
                "radius_obs_profile",
                "lambda_obs_xi2",
                "radius_obs_xi2",
                "z_obs_xi2",
            )
        }

        # outputs
        self.nc_zbin_lbin = None
        self.hbias_zbin_lbin = None
        self.gt_zbin_lbin_rbin = None
        self.xi2_zbin_lbin_rbin = None
        self.cov_nc_zbin_lbin = None
        self.cov_xi2_zbin_lbin_rbin = None

        # hardcoded quantities
        self.l_m_tab_sig_xi2 = [31, 51]
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        ################### INTERMIDIATE QUANTITIES ###################

        # computed in counts
        self._Plob_M_z = None
        self._dV_dzob = None
        self._Pltrue_M_z = None

        # computed in clustering
        self._window_radial = None
        self._volume_radial = None
        self._volume_zob = None
        self._one_over_n_lambdai_lambdaj = None
        self._Pk_lambdai_lambdaj = None

        ################### INTERNAL QUANTITIES ###################

        # volume element at the center of observed redshift bins
        self.dvdzdomega_z1z2 = derived_cosmology.dV_dzdO(
            self.background, self.integ_ztrue_arr, hubble_units=True
        )

        # hmf at the center of observed redshift bins
        self.dndm_z = self.hmfbias.dn_dm(self.integ_ztrue_arr, self.integ_mass_arr)
        self.bias_z = self.hmfbias.bias(
            self.integ_ztrue_arr, self.integ_mass_arr
        )  # only work for virial overdensity

    #####################################################################
    #
    # Core computation functions
    #
    #   These functions take explict arguments (and internal bins)
    #   to do the computations, they do not assign internal properties.
    #
    #####################################################################

    # ---------------
    # cluster counts
    # ---------------

    def _compute_volume_bin(self, z_bin, lambda_bin):
        """compute volume bin.
        Computes volume in a given richness redshift bin for cluster counts.

        Parameters
        ----------
        z_bin: int
            integer corresponding to redshift bin index
        lambda_bin: int
            integer corresponding to richness bin index

        Returns
        -------
        dV_dzob_bin: numpy.ndarray
            observed volume element dV/dz_ob
        """
        # computes volume in richness redshift bin for cluster counts

        z_tab = np.linspace(
            self.bins["z_obs_nc"].edges[z_bin],
            self.bins["z_obs_nc"].edges[z_bin + 1],
            self.z_tab_sig,
        )
        # P(zob|ztr)
        Pzob_z = simps(
            self.selectionfunction.P_zobs_z(
                z_tab,
                self.bins["lambda_obs_nc"].edges[lambda_bin],
                self.integ_ztrue_arr,
            ),
            x=z_tab,
            axis=0,
        )
        # observed volume element dV/dz_ob
        dV_dzob_bin = (
            self.dvdzdomega_z1z2 * Pzob_z * (self.area) * (np.pi**2.0 / 180.0**2.0)
        )
        # N(lob,zob)
        return dV_dzob_bin

    def _compute_Plob_M_z_bin(self, lambda_bin, Pltrue_M_z):
        """compute Plob_M_z_bin.
        Compute the probability of the observed richness 
        given true mass and redshift P(lob|M,ztr) for a richness bin.

        Parameters
        ----------
        lambda_bin : int
            integer corresponding to richness bin index
        Pltrue_M_z : numpy.ndarray
            probability of true richness given true mass and redshift  P(ltr|M,ztr)

        Returns
        -------
        numpy.ndarray
            P(lob|M,ztr)
        """
        # returns # P(lob|M,ztr) for a richness bin
        l_tab = np.geomspace(
            self.bins["lambda_obs_nc"].edges[lambda_bin],
            self.bins["lambda_obs_nc"].edges[lambda_bin + 1],
            self.l_m_tab_sig[lambda_bin],
        )
        # P(lob|ltr,ztr)
        Plob_l_z = simps(
            self.selectionfunction.P_lbdobs_lbd(
                self.integ_ztrue_arr, self.integ_lambda_arr, l_tab
            ),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF':
        #       Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.integ_ztrue_arr, self.integ_lambda_arr).T

        # P(lob|M,ztr)
        Plob_M_z = simps(
            Pltrue_M_z[:, :, :] * Plob_l_z[:, np.newaxis, :],
            x=self.integ_lambda_arr,
            axis=-1,
        )
        return Plob_M_z

    def _compute_counts_bin(self, dV_dzob_bin, n_lbdobs_z):
        """compute counts bin.
        Compute cluster counts in a single richness and redshift bin
        Performs integral over z_true of the the volume*n_lbdobs_z

        Parameters
        ----------
        dV_dzob_bin: numpy.ndarray
            volume element of bin
        n_lbdobs_z: numpy.ndarray
            number of clusters with observed richness and true redshift

        Returns
        -------
        numpy.ndarray
            counts in a richness redshift bin
        """
        # computes counts in a richness redshift bin
        return simps(n_lbdobs_z * dV_dzob_bin, x=self.integ_ztrue_arr, axis=0)

    def _compute_counts(self, Pltrue_M_z):
        """compute counts.
        Computes counts

        Parameters
        ----------
        Pltrue_M_z: numpy.ndarray
            Probability of lambda true given M_true and z_true

        Returns
        -------
        nc_zbin_lbin: numpy.ndarray
            number counts in richness and redshift bins
        _Plob_M_z: numpy.ndarray
            Probability of observed richness in redshift and true mass bins 
        _dV_dzob: numpy.ndarray
            array of bin volumes in redshift and richness
        """

        _n_lbdobs_z = np.zeros(self.bins["lambda_obs_nc"].size, dtype=list)

        # will be passed internal attributes
        _Plob_M_z = np.zeros(self.bins["lambda_obs_nc"].size, dtype=list)
        _dV_dzob = np.zeros(
            (self.bins["z_obs_nc"].size, self.bins["lambda_obs_nc"].size), dtype=list
        )

        # output
        nc_zbin_lbin = np.zeros(
            (self.bins["z_obs_nc"].size, self.bins["lambda_obs_nc"].size)
        )

        for lambda_bin in range(self.bins["lambda_obs_nc"].size):

            _Plob_M_z[lambda_bin] = self._compute_Plob_M_z_bin(lambda_bin, Pltrue_M_z)
            # N(lob,ztr)
            _n_lbdobs_z[lambda_bin] = simps(
                _Plob_M_z[lambda_bin] * self.dndm_z, x=self.integ_mass_arr, axis=1
            )

            for z_bin in range(self.bins["z_obs_nc"].size):

                _dV_dzob[z_bin, lambda_bin] = self._compute_volume_bin(
                    z_bin, lambda_bin
                )
                nc_zbin_lbin[z_bin, lambda_bin] = self._compute_counts_bin(
                    _dV_dzob[z_bin, lambda_bin], _n_lbdobs_z[lambda_bin]
                )
        return nc_zbin_lbin, _Plob_M_z, _dV_dzob

    # ----------
    # halo bias
    # ----------

    def _compute_bias(self, Plob_M_z, dV_dzob):
        """compute bias.
        Compute halo bias used in counts covariance in bins of redshift and richness

        Parameters
        ----------
        Plob_M_z : numpy.ndarray
            probability of observed richness given true mass and redshift
        dV_dzob : numpy.ndarray
            array of bin volumes in redshift and richness
            
        Returns
        -------
        numpy.ndarray
            halo bias in bins of z and lambda
        """

        hbias_zbin_lbin = np.zeros(
            (self.bins["z_obs_nc"].size, self.bins["lambda_obs_nc"].size)
        )

        for lambda_bin in range(self.bins["lambda_obs_nc"].size):

            # N(lob,ztr) * bias(lob,ztr)
            _b_n_lbdobs_z = simps(
                Plob_M_z[lambda_bin] * self.dndm_z * self.bias_z,
                x=self.integ_mass_arr,
                axis=1,
            )

            for z_bin in range(self.bins["z_obs_nc"].size):

                # N(lob,zob) * bias(lob,zob)
                hbias_zbin_lbin[z_bin, lambda_bin] = simps(
                    _b_n_lbdobs_z * dV_dzob[z_bin, lambda_bin],
                    x=self.integ_ztrue_arr,
                    axis=0,
                )

        return hbias_zbin_lbin

    # -------------------
    # cluster counts cov
    # -------------------

    def _compute_sab(self, z_obs_nc_mid):
        # initialization
        sab = np.zeros((self.bins["z_obs_nc"].size, self.bins["z_obs_nc"].size))
        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()
        # self.rint = np.zeros((self.bins["z_obs_nc"].size,len(self.integ_k_arr),L+1))

        # power spectrum at the center of observed redshift bins
        pk = self.halostatistics.matter_power_spectrum(z_obs_nc_mid, self.integ_k_arr)

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        photoz_corr0 = self.clustering.photoz_rsd_correction(z_obs_nc_mid, 0)[
            0
        ]  # can neglect richness dependence here
        pk *= photoz_corr0

        # richness-independent quantites that can be computed outside the lambda loop
        for z_bin in range(len(z_obs_nc_mid)):

            z_tab = np.linspace(
                self.bins["z_obs_nc"].edges[z_bin],
                self.bins["z_obs_nc"].edges[z_bin + 1],
                self.z_tab_sig,
            )

            sab[z_bin, : (z_bin + 1)] = (
                1
                / (2 * np.pi**2)
                * simps(
                    (self.integ_k_arr**2 * np.sqrt(pk[z_bin] * pk[: (z_bin + 1)]))
                    * self.covariance.cov_window(z_bin, z_tab, KL),
                    x=self.integ_k_arr,
                    axis=-1,
                )
            )
            sab[: (z_bin + 1), z_bin] = sab[z_bin, : (z_bin + 1)]

        return sab

    def _compute_counts_cov(self, hbias_zbin_lbin):
        """compute counts cov.
        computes theoretical covariance for cluster counts, including shot noise and sample covariance
        
        Parameters
        ----------
        hbias_zbin_lbin : numpy.ndarray
            halo bias, computed with compute_bias function

        Returns
        -------
        numpy.ndarray
            covariance array 
        """

        sab = self._compute_sab(
            _bin_midpoints(self.bins["z_obs_nc"].edges)  # mean observed redshift
        )

        # shot noise
        _shot_noise = (
            np.diag(self.nc_zbin_lbin.flatten())
            .reshape(
                *self.nc_zbin_lbin.shape,
                *self.nc_zbin_lbin.shape,
            )
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance
        cov_nc_zbin_lbin = _shot_noise + (
            hbias_zbin_lbin[np.newaxis, :, np.newaxis, :]
            * hbias_zbin_lbin[:, np.newaxis, :, np.newaxis]
            * sab[:, :, np.newaxis, np.newaxis]
        )

        return cov_nc_zbin_lbin

    # --------------
    # reduced shear
    # --------------

    def _compute_reduced_shear(self, nc_zbin_lbin, Plob_M_z, dV_dzob):
        """compute reduced shear.
        returns reduced shear array 

        Parameters
        ----------
        nc_zbin_lbin : numpy.ndarray
            cluster number counts
        Plob_M_z: numpy.ndarray
            probablity of observed richness given mass and redshift
        dV_dzob: numpy.ndarray
            volume element
        
        Returns
        -------
        numpy.ndarray
            reduced shear array
        """

        gt_zbin_lbin_rbin = np.zeros(
            (
                self.bins["z_obs_nc"].size,
                self.bins["lambda_obs_nc"].size,
                self.bins["radius_obs_profile"].size,
            )
        )

        for rad_bin in range(self.bins["radius_obs_profile"].size):
            excess_surface_mass_density = self.profile.excess_surface_mass_density(
                np.atleast_1d(self.bins["radius_obs_profile"].edges[rad_bin]),
                self.integ_ztrue_arr,
                self.integ_mass_arr,
                self.halo_concentration,
            )
            for lambda_bin in range(self.bins["lambda_obs_nc"].size):
                excesssurfacemassdensity = simps(
                    Plob_M_z[lambda_bin]
                    * self.dndm_z
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.integ_mass_arr,
                    axis=1,
                )

                for z_bin in range(self.bins["z_obs_nc"].size):
                    gt_zbin_lbin_rbin[z_bin, lambda_bin, rad_bin] = (
                        (1.0)
                        / nc_zbin_lbin[z_bin, lambda_bin]
                        * simps(
                            self.profile.m_sig_crit_m1(self.integ_ztrue_arr, z_bin)
                            * dV_dzob[z_bin, lambda_bin]
                            * excesssurfacemassdensity,
                            x=self.integ_ztrue_arr,
                        )
                    )
        return gt_zbin_lbin_rbin

    # -----------
    # clustering
    # -----------

    def _compute_pk_ir_resummation(self, Pltrue_M_z):

        _volume_zob = np.zeros(self.bins["z_obs_xi2"].size)
        _one_over_n_lambdai_lambdaj = np.zeros(
            (
                (
                    self.bins["z_obs_xi2"].size,
                    self.bins["lambda_obs_xi2"].size,
                    self.bins["lambda_obs_xi2"].size,
                )
            )
        )

        lambda_obs_xi2_mid = _bin_midpoints(self.bins["lambda_obs_xi2"].edges)

        sqrt_Pk_zbin_lbin = np.zeros(
            (
                self.bins["z_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                len(self.integ_k_arr),
            )
        )

        ############
        #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
        ############
        pk_IR = self.halostatistics.matter_power_spectrum(
            self.integ_ztrue_arr, self.integ_k_arr
        )

        # LOOP OVER CLUSTERING RICHNESS BINS
        for lambda_bin in range(self.bins["lambda_obs_xi2"].size):

            l_tab = np.geomspace(
                self.bins["lambda_obs_xi2"].edges[lambda_bin],
                self.bins["lambda_obs_xi2"].edges[lambda_bin + 1],
                self.l_m_tab_sig_xi2[lambda_bin],
            )

            ## recompute quantities that depend on lambda_obs

            # P(lob|ltr,ztr)
            Plob_l_z_xi2 = simps(
                self.selectionfunction.P_lbdobs_lbd(
                    self.integ_ztrue_arr, self.integ_lambda_arr, l_tab
                ),
                x=l_tab,
                axis=-1,
            )

            # P(lob|M,ztr)
            Plob_M_z_xi2 = simps(
                Pltrue_M_z[:, :, :] * Plob_l_z_xi2[:, np.newaxis, :],
                x=self.integ_lambda_arr,
                axis=-1,
            )

            # n(lob,ztr)
            n_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2 * self.dndm_z, x=self.integ_mass_arr, axis=1
            )

            # n(lob,ztr) * b(lob,zob)
            n_b_lbdobs_z_xi2 = simps(
                Plob_M_z_xi2 * self.dndm_z * self.bias_z, x=self.integ_mass_arr, axis=1
            )

            # effective halo bias
            b_eff = (n_b_lbdobs_z_xi2 / n_lbdobs_z_xi2)[:, np.newaxis]

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.clustering.photoz_rsd_correction(
                    self.integ_ztrue_arr, lambda_obs_xi2_mid[lambda_bin]
                )
            )
            pk_halo = pk_IR * (
                b_eff**2 * photoz_corr0 + b_eff * photoz_corr1 + photoz_corr2
            )

            for z_bin in range(self.bins["z_obs_xi2"].size):

                z_tab = np.linspace(
                    self.bins["z_obs_xi2"].edges[z_bin],
                    self.bins["z_obs_xi2"].edges[z_bin + 1],
                    self.z_tab_sig,
                )

                # P(zob|ztr)
                Pzob_z = simps(
                    self.selectionfunction.P_zobs_z(
                        z_tab,
                        self.bins["lambda_obs_xi2"].edges[lambda_bin],
                        self.integ_ztrue_arr,
                    ),
                    x=z_tab,
                    axis=0,
                )

                # observed volume element dV/dz_ob
                dV_dzob = (
                    self.dvdzdomega_z1z2
                    * Pzob_z
                    * (self.area)
                    * (np.pi**2.0 / 180.0**2.0)
                )

                # volume of the observed redshift slice
                _volume_zob[z_bin] = simps(dV_dzob, x=self.integ_ztrue_arr, axis=0)

                # normalization factor
                N_int_lbdobs_z_xi2 = simps(
                    dV_dzob * n_lbdobs_z_xi2, x=self.integ_ztrue_arr, axis=0
                )

                # power spectrum and shot-noise terms
                sqrt_Pk_zbin_lbin[z_bin, lambda_bin, :] = (
                    simps(
                        (dV_dzob * n_lbdobs_z_xi2)[:, np.newaxis] * np.sqrt(pk_halo),
                        x=self.integ_ztrue_arr,
                        axis=0,
                    )
                    / N_int_lbdobs_z_xi2
                )

                _one_over_n_lambdai_lambdaj[z_bin, lambda_bin, lambda_bin] = (
                    _volume_zob[z_bin] / N_int_lbdobs_z_xi2
                )

        # cross Pk and shot-noise in two richness bins
        _Pk_lambdai_lambdaj = (
            sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )  # dim = [nz,nl,nl,nk]
        _one_over_n_lambdai_lambdaj = _one_over_n_lambdai_lambdaj[
            :, :, :, np.newaxis
        ]  # dim = [nz,nl,nl,nk]

        return _Pk_lambdai_lambdaj, _volume_zob, _one_over_n_lambdai_lambdaj

    def _compute_xi2(self, window_radial, Pk_lambdai_lambdaj):

        # compute 2point correlation function
        # dim = [nz,nl,nl,nr]
        xi2_zbin_lbin_rbin_buf = simps(
            (
                self.integ_k_arr**2.0
                / (2.0 * np.pi**2)
                * window_radial[:, np.newaxis, np.newaxis, :, :]
                * Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            ),
            x=self.integ_k_arr,
            axis=-1,
        )

        # xi(lambda_i, lambda_j) = xi(lambda_j, lambda_i),  so reshape and keep only one of them

        triangle_indexes = np.triu_indices(self.bins["lambda_obs_xi2"].size)
        return xi2_zbin_lbin_rbin_buf[:, triangle_indexes[0], triangle_indexes[1], :]

    # ----------------------
    # clustering covariance
    # ----------------------

    def _compute_alpha_beta(
        self,
        alpha_cov_xi2,
        beta_cov_xi2,
        Pk_lambdai_lambdaj,
        one_over_n_lambdai_lambdaj,
    ):
        # combine and reshape
        alpha_ij = (1 + alpha_cov_xi2[:, :, np.newaxis, np.newaxis]) * (
            1 + alpha_cov_xi2[:, np.newaxis, :, np.newaxis]
        )
        beta_ij = (
            beta_cov_xi2[:, :, np.newaxis, np.newaxis]
            * beta_cov_xi2[:, np.newaxis, :, np.newaxis]
        )

        beta_pk_ij = beta_ij * Pk_lambdai_lambdaj
        alpha_n_ij = alpha_ij * one_over_n_lambdai_lambdaj

        return alpha_n_ij, beta_pk_ij

    def _compute_xi2_cov(
        self,
        Pk_lambdai_lambdaj,
        one_over_n_lambdai_lambdaj,
        window_radial,
        volume_radial,
        volume_zob,
    ):
        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0,beta=1,gamma=0 (see Euclid Collaboration:
        #    Fumagalli et al. 2022)
        #    cov_g, cov_ng are TWO TERMS OF EQ. 73
        _alpha_cov_xi2 = np.zeros(
            (self.bins["z_obs_xi2"].size, self.bins["lambda_obs_xi2"].size)
        )
        _beta_cov_xi2 = np.ones(
            (self.bins["z_obs_xi2"].size, self.bins["lambda_obs_xi2"].size)
        )
        gamma_cov_xi2 = np.zeros(
            (self.bins["z_obs_xi2"].size, self.bins["lambda_obs_xi2"].size)
        )

        # compute nuisance parameters
        alpha_n_ij, beta_pk_ij = self._compute_alpha_beta(
            _alpha_cov_xi2,
            _beta_cov_xi2,
            Pk_lambdai_lambdaj,
            one_over_n_lambdai_lambdaj,
        )

        # internal attributes
        _cov_g = np.zeros(
            (
                self.bins["z_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["radius_obs_xi2"].size,
                self.bins["radius_obs_xi2"].size,
            )
        )
        _cov_ng = np.zeros(
            (
                self.bins["z_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["lambda_obs_xi2"].size,
                self.bins["radius_obs_xi2"].size,
                self.bins["radius_obs_xi2"].size,
            )
        )

        # define cluster clustering bin numbers for loops
        z_bin_numbers = range(self.bins["z_obs_xi2"].size)
        lambda_bin_numbers = range(self.bins["lambda_obs_xi2"].size)
        rad_bin_numbers = range(self.bins["radius_obs_xi2"].size)

        # note: this could be reduced to compute only half of the matrix
        for lambda_bin_i in lambda_bin_numbers:
            for lambda_bin_j in lambda_bin_numbers:
                for rad_bin in rad_bin_numbers:

                    _cov_ng[
                        :,
                        lambda_bin_i,
                        lambda_bin_j,
                        lambda_bin_i,
                        lambda_bin_j,
                        rad_bin,
                        rad_bin,
                    ] = (
                        simps(
                            self.integ_k_arr**2.0
                            / (2.0 * np.pi**2.0)
                            * window_radial[:, rad_bin, :]
                            * beta_pk_ij[:, lambda_bin_i, lambda_bin_j, :],
                            x=self.integ_k_arr,
                        )
                        * (1 + gamma_cov_xi2[:, lambda_bin_i])
                        * one_over_n_lambdai_lambdaj[:, lambda_bin_i, lambda_bin_i, 0]
                        * (1 + gamma_cov_xi2[:, lambda_bin_j])
                        * one_over_n_lambdai_lambdaj[:, lambda_bin_j, lambda_bin_j, 0]
                        / volume_radial[:, rad_bin]
                    )

                    for lambda_bin_k in lambda_bin_numbers:
                        for lambda_bin_h in lambda_bin_numbers:

                            # gaussian term
                            _cov_g[
                                :,
                                lambda_bin_i,
                                lambda_bin_j,
                                lambda_bin_k,
                                lambda_bin_h,
                                :,
                                :,
                            ] = simps(
                                self.integ_k_arr**2.0
                                / (2.0 * np.pi**2.0)
                                * window_radial[:, np.newaxis, :, :]
                                * window_radial[:, :, np.newaxis, :]
                                * (beta_pk_ij + alpha_n_ij)[
                                    :,
                                    lambda_bin_i,
                                    lambda_bin_k,
                                    np.newaxis,
                                    np.newaxis,
                                    :,
                                ]
                                * (beta_pk_ij + alpha_n_ij)[
                                    :,
                                    lambda_bin_j,
                                    lambda_bin_h,
                                    np.newaxis,
                                    np.newaxis,
                                    :,
                                ],
                                x=self.integ_k_arr,
                                axis=-1,
                            )

        # Compute the covariance
        cov_xi2_zbin_lbin_rbin = (
            (_cov_g + _cov_ng)
            + (_cov_g + _cov_ng).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_xi2 bins
            )
        ) / volume_zob[
            :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_i, lambda_j, lambda_k, lambda_l) = cov_xi(lambda_j, lambda_i, lambda_l, lambda_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(self.bins["lambda_obs_xi2"].size)
        # simplify first pair
        cov_xi2_zbin_lbin_rbin = cov_xi2_zbin_lbin_rbin[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        cov_xi2_zbin_lbin_rbin = cov_xi2_zbin_lbin_rbin[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (nz, nl_red, nl_red, nrad, nrad),
        # make it (nz, nz, nl_red, nl_red, nrad, nrad), being diagonal in (nz, nz)
        # """
        cov_xi2_zbin_lbin_rbin = (
            np.diag(np.ones(self.bins["z_obs_xi2"].size))[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * cov_xi2_zbin_lbin_rbin
        )
        return cov_xi2_zbin_lbin_rbin

    #####################################################################
    #
    # Assignemnt functions
    #
    #   These functions use the core computation functions results and
    #   assign them to internal properties.
    #
    #####################################################################

    def _missing_attributes(self, *attr):
        """Checks if any of the provided attributes it None"""
        return any(_attr is None for _attr in attr)

    def compute_counts(self):

        # P(ltrM,ztr), this quantity is also used by cluster clustering
        self._Pltrue_M_z = self.selectionfunction.P_lnlbd(
            self.integ_ztrue_arr, self.integ_mass_arr, self.integ_lambda_arr
        )

        self.nc_zbin_lbin, self._Plob_M_z, self._dV_dzob = self._compute_counts(
            self._Pltrue_M_z
        )

    def compute_bias(self):

        # check precomputed attributes
        if self._missing_attributes(self._Plob_M_z, self._dV_dzob):
            raise ValueError("Run compute_counts first!")

        self.hbias_zbin_lbin = self._compute_bias(self._Plob_M_z, self._dV_dzob)

    def compute_counts_cov(self):

        # check precomputed attributes
        if self._missing_attributes(self.hbias_zbin_lbin):
            raise ValueError("Run compute_bias first!")

        self.cov_nc_zbin_lbin = self._compute_counts_cov(self.hbias_zbin_lbin)

    def compute_reduced_shear(self):

        # check precomputed attributes
        if self._missing_attributes(self.nc_zbin_lbin, self._Plob_M_z, self._dV_dzob):
            raise ValueError("Run compute_counts first!")

        self.gt_zbin_lbin_rbin = self._compute_reduced_shear(
            self.nc_zbin_lbin, self._Plob_M_z, self._dV_dzob
        )

    def compute_xi2(self):

        # check precomputed attributes
        if self._missing_attributes(self._Pltrue_M_z):
            raise ValueError("Run compute_counts first!")

        # this is never used
        ###integ_zbin_lbin  = np.zeros(((self.bins["z_obs_xi2"].size, self.bins["lambda_obs_xi2"].size, len(self.integ_k_arr))))

        # matter power spectrum + IR resummation
        self._Pk_lambdai_lambdaj, self._volume_zob, self._one_over_n_lambdai_lambdaj = (
            self._compute_pk_ir_resummation(self._Pltrue_M_z)
        )

        # spherical shell window function W(R,K) and volume of the shell (compute once outside the redshift loop)
        # these are used by covariance
        self._window_radial, self._volume_radial = self.clustering.WF_ra(
            _bin_midpoints(self.bins["z_obs_xi2"].edges),  # z_obs_xi2_mid
            self.bins["radius_obs_xi2"].edges,
        )

        self.xi2_zbin_lbin_rbin = self._compute_xi2(
            self._window_radial, self._Pk_lambdai_lambdaj
        )

    def compute_xi2_cov(self):

        # check precomputed attributes
        if self._missing_attributes(
            self._Pk_lambdai_lambdaj,
            self._one_over_n_lambdai_lambdaj,
            self._window_radial,
            self._volume_radial,
            self._volume_zob,
        ):
            raise ValueError("Run compute_xi2 first!")

        self.cov_xi2_zbin_lbin_rbin = self._compute_xi2_cov(
            self._Pk_lambdai_lambdaj,
            self._one_over_n_lambdai_lambdaj,
            self._window_radial,
            self._volume_radial,
            self._volume_zob,
        )

    def compute_all_quantities(
        self,
        like_selection: str = "CC_CWL_Cxi2",
        cov_selection: str = "covCC_covCxi2",
        #        external_richness_selection_function: str = 'non_CG_ESF',
        z_obs_nc_edges: np.ndarray = None,
        lambda_obs_nc_edges: np.ndarray = None,
        radius_obs_profile_edges: np.ndarray = None,
        lambda_obs_xi2_edges: np.ndarray = None,
        radius_obs_xi2_edges: np.ndarray = None,
        z_obs_xi2_edges: np.ndarray = None,
    ):

        # assing edges

        self.bins["z_obs_nc"].edges = z_obs_nc_edges
        self.bins["lambda_obs_nc"].edges = lambda_obs_nc_edges
        self.bins["radius_obs_profile"].edges = radius_obs_profile_edges
        self.bins["lambda_obs_xi2"].edges = lambda_obs_xi2_edges
        self.bins["radius_obs_xi2"].edges = radius_obs_xi2_edges
        self.bins["z_obs_xi2"].edges = z_obs_xi2_edges

        # main function

        if "CC" in like_selection:
            self.compute_counts()
            if "covCC" in cov_selection:
                self.compute_bias()
                self.compute_counts_cov()

        if "CWL" in like_selection:
            self.compute_reduced_shear()

        if "Cxi2" in like_selection:
            self.compute_xi2()
            if "covCxi2" in cov_selection:
                self.compute_xi2_cov()


class Bin:
    def __init__(self, name, edges=None):
        self.name = name
        self.edges = edges

    @property
    def size(self):
        # observed redshift bins for number counts and weak lensing
        if self.edges is None:
            raise ValueError(f"bins for {self.name} not set")
        return len(self.edges) - 1


def _bin_midpoints(edges):
    """bin midpoints.

    computs bin midpoints given numpy array of edges

    Parameters
    ----------
    edges : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return 0.5 * (edges[1:] + edges[:-1])

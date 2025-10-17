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
        z_obs_NC_edges: np.ndarray,
        Lambda_obs_NC_edges: np.ndarray,
        Rad_obs_edges: np.ndarray,
        Lambda_obs_Cxi2_edges: np.ndarray,
        Rad_obs_Cxi2_edges: np.ndarray,
        z_obs_Cxi2_edges: np.ndarray,
        halo_concentration: float,
        k: np.ndarray,
        Mass: np.ndarray,
        Lambda: np.ndarray,
        z: np.ndarray,
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

        # values
        self.area = area

        self.halo_concentration = halo_concentration

        # edges
        self.z_obs_NC_edges = z_obs_NC_edges
        self.Lambda_obs_NC_edges = Lambda_obs_NC_edges
        self.Rad_obs_edges = Rad_obs_edges
        self.Lambda_obs_Cxi2_edges = Lambda_obs_Cxi2_edges
        self.Rad_obs_Cxi2_edges = Rad_obs_Cxi2_edges
        self.z_obs_Cxi2_edges = z_obs_Cxi2_edges

        # integration variables
        self.k = k  # k array
        self.Mass = Mass  # mass array in Msun h^-1
        self.Lambda = Lambda  # true richness array
        self.z = z  # true redshift array

        ################### SELECTION FUNCTION ###################

        # selections (???)
        self.l_m_tab_sig_Cxi2 = [31, 51]
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        ################### OBSERVED BINS ###################

        # observed redshift bins for number counts and weak lensing
        # self.z_obs_NC_edges = z_obs_NC_edges #self.theory['obs_specifications']['CG']['z_obs_NC_edges']
        self.z_obs_NC_div = len(self.z_obs_NC_edges) - 1

        # observed richness bins for number counts and weak lensing
        # self.Lambda_obs_NC_edges = Lambda_obs_NC_edges #self.theory['obs_specifications']['CG']['Lambda_obs_NC_edges']
        self.Lambda_obs_NC_div = len(self.Lambda_obs_NC_edges) - 1

        # observed radial separation bins for weak lensing
        # self.Rad_obs_edges = Rad_obs_edges #self.theory['obs_specifications']['CG']['Rad_obs_edges']
        self.Rad_obs_div = len(self.Rad_obs_edges) - 1

        # observed richness bins for clustering
        # self.Lambda_obs_Cxi2_edges = Lambda_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['Lambda_obs_Cxi2_edges']
        self.Lambda_obs_Cxi2_div = len(self.Lambda_obs_Cxi2_edges) - 1

        # observed radial separation bins for clustering
        # self.Rad_obs_Cxi2_edges = Rad_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['Rad_obs_Cxi2_edges']
        self.Rad_obs_Cxi2_div = len(self.Rad_obs_Cxi2_edges) - 1

        # observed redshift bins for clustering
        # self.z_obs_Cxi2_edges = z_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['z_obs_Cxi2_edges']
        self.z_obs_Cxi2_div = len(self.z_obs_Cxi2_edges) - 1

        ################### INTERMIDIATE QUANTITIES ###################

        # computed in counts
        self._Plob_M_z = None
        self._dV_dzob = None
        self._n_lbdobs_z = None
        self._Pl_M_z = None

        # computed in counts covariance
        self._Nb_zbin_Lbin = None
        self._b_n_lbdobs_z = None
        self._shot_noise = None

        # computed in clustering
        self._W_rad = None
        self._V_rad = None
        self._V_zob = None
        self._sqrt_Pk_zbin_lbin = None
        self._one_over_n_lambdai_lambdaj = None
        self._Pk_lambdai_lambdaj = None

        # cluster clustering cov
        self._alpha_cov_Cxi2 = None
        self._beta_cov_Cxi2 = None
        self._gamma_cov_Cxi2 = None
        self._cov_g = None
        self._cov_ng = None

        ################### INTERNAL QUANTITIES ###################

        # volume element at the center of observed redshift bins
        self.dvdzdomega_z1z2 = derived_cosmology.dV_dzdO(
            self.background, self.z, hubble_units=True
        )

        # hmf at the center of observed redshift bins
        self.dndm_z = self.hmfbias.dn_dm(self.z, self.Mass)
        self.bias_z = self.hmfbias.bias(
            self.z, self.Mass
        )  # only work for virial overdensity

        ################### OUTPUTS ###################
        self.N_zbin_Lbin = None
        self.g_zbin_Lbin_Rbin = None
        self.Cxi2_zbin_Lbin_Rbin = None
        self.cov_NC_zbin_Lbin = None
        self.cov_Cxi2_zbin_Lbin_Rbin = None

    def _missing_attributes(self, *attr):
        """Checks if any of the provided attributes it None"""
        return any(_attr is None for _attr in attr)

    ################
    # cluster counts
    ################

    def _volume_bin(self, z_bin, lambda_bin):
        # computes volume in richness redshift bin for cluster counts

        z_tab = np.linspace(
            self.z_obs_NC_edges[z_bin], self.z_obs_NC_edges[z_bin + 1], self.z_tab_sig
        )
        # P(zob|ztr)
        Pzob_z = simps(
            self.selectionfunction.P_zobs_z(
                z_tab, self.Lambda_obs_NC_edges[lambda_bin], self.z
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

    def _Plob_M_z_bin(self, lambda_bin):
        # returns # P(lob|M,ztr) for a richness bin
        l_tab = np.geomspace(
            self.Lambda_obs_NC_edges[lambda_bin],
            self.Lambda_obs_NC_edges[lambda_bin + 1],
            self.l_m_tab_sig[lambda_bin],
        )
        # P(lob|ltr,ztr)
        Plob_l_z = simps(
            self.selectionfunction.P_lbdobs_lbd(self.z, self.Lambda, l_tab),
            x=l_tab,
            axis=-1,
        )
        #       if external_richness_selection_function == 'CG_ESF':
        #       Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.z, self.Lambda).T

        # P(lob|M,ztr)
        Plob_M_z = simps(
            self._Pl_M_z[:, :, :] * Plob_l_z[:, np.newaxis, :], x=self.Lambda, axis=-1
        )
        return Plob_M_z

    def _counts_bin(self, dV_dzob_bin, n_lbdobs_z):
        # computes counts in a richness redshift bin
        return simps(n_lbdobs_z * dV_dzob_bin, x=self.z, axis=0)

    def _init_counts_arrays(self):
        # internal attributes
        self._Plob_M_z = np.zeros(self.Lambda_obs_NC_div, dtype=list)
        self._dV_dzob = np.zeros(
            (self.z_obs_NC_div, self.Lambda_obs_NC_div), dtype=list
        )
        self._n_lbdobs_z = np.zeros(self.Lambda_obs_NC_div, dtype=list)
        # P(ltrM,ztr), this quantity is also used by cluster clustering
        self._Pl_M_z = self.selectionfunction.P_lnlbd(self.z, self.Mass, self.Lambda)

        # output
        self.N_zbin_Lbin = np.zeros((self.z_obs_NC_div, self.Lambda_obs_NC_div))

    def compute_counts(self):

        # prepare internal attributes
        self._init_counts_arrays()

        for lambda_bin in range(self.Lambda_obs_NC_div):

            self._Plob_M_z[lambda_bin] = self._Plob_M_z_bin(lambda_bin)
            # N(lob,ztr)
            self._n_lbdobs_z[lambda_bin] = simps(
                self._Plob_M_z[lambda_bin] * self.dndm_z, x=self.Mass, axis=1
            )

            for z_bin in range(self.z_obs_NC_div):

                self._dV_dzob[z_bin, lambda_bin] = self._volume_bin(z_bin, lambda_bin)
                self.N_zbin_Lbin[z_bin, lambda_bin] = self._counts_bin(
                    self._dV_dzob[z_bin, lambda_bin], self._n_lbdobs_z[lambda_bin]
                )

    #####################
    # cluster counts bias
    #####################

    def _init_bias_arrays(self):
        # internal attributes
        self._Nb_zbin_Lbin = np.zeros((self.z_obs_NC_div, self.Lambda_obs_NC_div))
        self._b_n_lbdobs_z = np.zeros(self.Lambda_obs_NC_div, dtype=list)

    def _compute_bias(self, Plob_M_z, dV_dzob):

        for lambda_bin in range(self.Lambda_obs_NC_div):

            # N(lob,ztr) * bias(lob,ztr)
            self._b_n_lbdobs_z[lambda_bin] = simps(
                Plob_M_z[lambda_bin] * self.dndm_z * self.bias_z,
                x=self.Mass,
                axis=1,
            )

            for z_bin in range(self.z_obs_NC_div):

                # N(lob,zob) * bias(lob,zob)
                self._Nb_zbin_Lbin[z_bin, lambda_bin] = simps(
                    self._b_n_lbdobs_z[lambda_bin] * dV_dzob[z_bin, lambda_bin],
                    x=self.z,
                    axis=0,
                )

    def compute_bias(self):

        # check precomputed attributes
        if self._missing_attributes(self._Plob_M_z, self._dV_dzob):
            raise ValueError("Run compute_counts first!")

        # prepare internal attributes
        self._init_bias_arrays()

        self._compute_bias(self._Plob_M_z, self._dV_dzob)

    ####################
    # cluster counts cov
    ####################

    def _compute_sab(self, z_obs_NC_mid):
        # initialization
        sab = np.zeros((self.z_obs_NC_div, self.z_obs_NC_div))
        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()
        # self.rint = np.zeros((self.z_obs_NC_div,len(self.k),L+1))

        # power spectrum at the center of observed redshift bins
        pk = self.halostatistics.matter_power_spectrum(z_obs_NC_mid, self.k)

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        photoz_corr0 = self.clustering.photoz_rsd_correction(z_obs_NC_mid, 0)[
            0
        ]  # can neglect richness dependence here
        pk *= photoz_corr0

        # richness-independent quantites that can be computed outside the lambda loop
        for z_bin in range(len(z_obs_NC_mid)):

            z_tab = np.linspace(
                self.z_obs_NC_edges[z_bin],
                self.z_obs_NC_edges[z_bin + 1],
                self.z_tab_sig,
            )

            sab[z_bin, : (z_bin + 1)] = (
                1
                / (2 * np.pi**2)
                * simps(
                    (self.k**2 * np.sqrt(pk[z_bin] * pk[: (z_bin + 1)]))
                    * self.covariance.cov_window(z_bin, z_tab, KL),
                    x=self.k,
                    axis=-1,
                )
            )
            sab[: (z_bin + 1), z_bin] = sab[z_bin, : (z_bin + 1)]

        return sab

    def _init_counts_cov_arrays(self):
        # output
        self.cov_NC_zbin_Lbin = np.zeros(
            (
                self.z_obs_NC_div,
                self.z_obs_NC_div,
                self.Lambda_obs_NC_div,
                self.Lambda_obs_NC_div,
            )
        )

    def _compute_counts_cov(self, Nb_zbin_Lbin):

        sab = self._compute_sab(
            _bin_midpoints(self.z_obs_NC_edges)  # mean observed redshift
        )

        # shut noise
        self._shot_noise = (
            np.diag(self.N_zbin_Lbin.flatten())
            .reshape(
                self.z_obs_NC_div,
                self.Lambda_obs_NC_div,
                self.z_obs_NC_div,
                self.Lambda_obs_NC_div,
            )
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance
        self.cov_NC_zbin_Lbin = self._shot_noise + (
            Nb_zbin_Lbin.reshape(1, self.z_obs_NC_div, 1, self.Lambda_obs_NC_div)
            * Nb_zbin_Lbin.reshape(self.z_obs_NC_div, 1, self.Lambda_obs_NC_div, 1)
            * sab.reshape(self.z_obs_NC_div, self.z_obs_NC_div, 1, 1)
        )

    def compute_counts_cov(self):

        # check precomputed attributes
        if self._missing_attributes(self._Nb_zbin_Lbin):
            raise ValueError("Run compute_bias first!")

        # prepare internal attributes
        self._init_counts_cov_arrays()

        self._compute_counts_cov(self._Nb_zbin_Lbin)

        return

    ###############
    # reduced shear
    ###############

    def _init_reduced_shear_arrays(self):
        # output
        self.g_zbin_Lbin_Rbin = np.zeros(
            (self.z_obs_NC_div, self.Lambda_obs_NC_div, self.Rad_obs_div)
        )

    def compute_reduced_shear(self, N_zbin_Lbin):

        # check precomputed attributes
        if self._missing_attributes(self._Plob_M_z, self._dV_dzob):
            raise ValueError("Run compute_counts first!")

        # prepare internal attributes
        self._init_reduced_shear_arrays()

        for rad_bin in range(self.Rad_obs_div):
            excess_surface_mass_density = self.profile.excess_surface_mass_density(
                np.atleast_1d(self.Rad_obs_edges[rad_bin]),
                self.z,
                self.Mass,
                self.halo_concentration,
            )
            for lambda_bin in range(self.Lambda_obs_NC_div):
                excesssurfacemassdensity = simps(
                    self._Plob_M_z[lambda_bin]
                    * self.dndm_z
                    * np.squeeze(excess_surface_mass_density, axis=2),
                    x=self.Mass,
                    axis=1,
                )

                for z_bin in range(self.z_obs_NC_div):
                    self.g_zbin_Lbin_Rbin[z_bin, lambda_bin, rad_bin] = (
                        (1.0)
                        / N_zbin_Lbin[z_bin, lambda_bin]
                        * simps(
                            self.profile.m_sig_crit_m1(self.z, z_bin)
                            * self._dV_dzob[z_bin, lambda_bin]
                            * excesssurfacemassdensity,
                            x=self.z,
                        )
                    )

    ############
    # clustering
    ############

    def _init_Cxi2_arrays(self):
        # internal attributes
        self._V_zob = np.zeros(self.z_obs_Cxi2_div)
        self._sqrt_Pk_zbin_lbin = np.zeros(
            (self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, len(self.k))
        )
        self._one_over_n_lambdai_lambdaj = np.zeros(
            ((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, self.Lambda_obs_Cxi2_div))
        )
        # output
        self.Cxi2_zbin_Lbin_Rbin = np.zeros(
            (
                self.z_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div + 1,
                self.Rad_obs_Cxi2_div,
            )
        )

    def compute_Cxi2(self):

        ### !!!! note that the final number of richness bins is NL=nl+1 ONLY if we have two richness bins,
        ### if nl>2, the effective number of richness bins is NL=factorial(nl)//(factorial(nl-2)*factorial(2)) + nl
        ### this makes the reshape of the matrix more complex. Since we plan to use only two bins, for the moment it is not implemented.

        # check precomputed attributes
        if self._missing_attributes(self._Pl_M_z):
            raise ValueError("Run compute_counts first!")

        # prepare internal attributes
        self._init_Cxi2_arrays()

        # this is never used
        ###integ_zbin_lbin  = np.zeros(((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, len(self.k))))

        # clustering bins
        z_obs_Cxi2_mid = _bin_midpoints(self.z_obs_Cxi2_edges)
        Lambda_obs_Cxi2_mid = _bin_midpoints(self.Lambda_obs_Cxi2_edges)

        # spherical shell window function W(R,K) and volume of the shell (compute once outside the redshift loop)
        # these are used by covariance
        self._W_rad, self._V_rad = self.clustering.WF_ra(
            z_obs_Cxi2_mid, self.Rad_obs_Cxi2_edges
        )

        # matter power spectrum + IR resummation
        ############
        #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
        ############
        pk_IR = self.halostatistics.matter_power_spectrum(self.z, self.k)

        # LOOP OVER CLUSTERING RICHNESS BINS
        for lambda_bin in range(self.Lambda_obs_Cxi2_div):

            l_tab = np.geomspace(
                self.Lambda_obs_Cxi2_edges[lambda_bin],
                self.Lambda_obs_Cxi2_edges[lambda_bin + 1],
                self.l_m_tab_sig_Cxi2[lambda_bin],
            )

            ## recompute quantities that depend on lambda_obs

            # P(lob|ltr,ztr)
            Plob_l_z_Cxi2 = simps(
                self.selectionfunction.P_lbdobs_lbd(self.z, self.Lambda, l_tab),
                x=l_tab,
                axis=-1,
            )

            # P(lob|M,ztr)
            Plob_M_z_Cxi2 = simps(
                self._Pl_M_z[:, :, :] * Plob_l_z_Cxi2[:, np.newaxis, :],
                x=self.Lambda,
                axis=-1,
            )

            # n(lob,ztr)
            n_lbdobs_z_Cxi2 = simps(Plob_M_z_Cxi2 * self.dndm_z, x=self.Mass, axis=1)

            # n(lob,ztr) * b(lob,zob)
            n_b_lbdobs_z_Cxi2 = simps(
                Plob_M_z_Cxi2 * self.dndm_z * self.bias_z, x=self.Mass, axis=1
            )

            # effective halo bias
            b_eff = (n_b_lbdobs_z_Cxi2 / n_lbdobs_z_Cxi2)[:, np.newaxis]

            # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
            photoz_corr0, photoz_corr1, photoz_corr2 = (
                self.clustering.photoz_rsd_correction(
                    self.z, Lambda_obs_Cxi2_mid[lambda_bin]
                )
            )
            pk_halo = pk_IR * (
                b_eff**2 * photoz_corr0 + b_eff * photoz_corr1 + photoz_corr2
            )

            for z_bin in range(self.z_obs_Cxi2_div):

                z_tab = np.linspace(
                    self.z_obs_Cxi2_edges[z_bin],
                    self.z_obs_Cxi2_edges[z_bin + 1],
                    self.z_tab_sig,
                )

                # P(zob|ztr)
                Pzob_z = simps(
                    self.selectionfunction.P_zobs_z(
                        z_tab, self.Lambda_obs_Cxi2_edges[lambda_bin], self.z
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
                self._V_zob[z_bin] = simps(dV_dzob, x=self.z, axis=0)

                # normalization factor
                N_int_lbdobs_z_Cxi2 = simps(dV_dzob * n_lbdobs_z_Cxi2, x=self.z, axis=0)

                # power spectrum and shot-noise terms
                self._sqrt_Pk_zbin_lbin[z_bin, lambda_bin, :] = (
                    simps(
                        (dV_dzob * n_lbdobs_z_Cxi2)[:, np.newaxis] * np.sqrt(pk_halo),
                        x=self.z,
                        axis=0,
                    )
                    / N_int_lbdobs_z_Cxi2
                )

                self._one_over_n_lambdai_lambdaj[z_bin, lambda_bin, lambda_bin] = (
                    self._V_zob[z_bin] / N_int_lbdobs_z_Cxi2
                )

        # cross Pk and shot-noise in two richness bins
        self._Pk_lambdai_lambdaj = (
            self._sqrt_Pk_zbin_lbin[:, :, np.newaxis, :]
            * self._sqrt_Pk_zbin_lbin[:, np.newaxis, :, :]
        )  # dim = [nz,nl,nl,nk]
        self._one_over_n_lambdai_lambdaj = self._one_over_n_lambdai_lambdaj[
            :, :, :, np.newaxis
        ]  # dim = [nz,nl,nl,nk]

        # compute 2point correlation function
        # dim = [nz,nl,nl,nr]
        Cxi2_zbin_Lbin_Rbin_buf = simps(
            (
                self.k**2.0
                / (2.0 * np.pi**2)
                * self._W_rad[:, np.newaxis, np.newaxis, :, :]
                * self._Pk_lambdai_lambdaj[:, :, :, np.newaxis, :]
            ),
            x=self.k,
            axis=-1,
        )

        # xi(lambda_i,lambda_j) = xi(lambda_j,lambda_i), so reshape and keep only one of them
        for z_bin in range(self.z_obs_Cxi2_div):
            for rad_bin in range(self.Rad_obs_Cxi2_div):
                self.Cxi2_zbin_Lbin_Rbin[z_bin, :, rad_bin] = Cxi2_zbin_Lbin_Rbin_buf[
                    z_bin, :, :, rad_bin
                ][np.triu_indices(self.Lambda_obs_Cxi2_div)]

    #######################
    # clustering covariance
    #######################

    def _compute_alpha_beta(self):
        # combine and reshape
        alpha_ij = (1 + self._alpha_cov_Cxi2[:, :, np.newaxis, np.newaxis]) * (
            1 + self._alpha_cov_Cxi2[:, np.newaxis, :, np.newaxis]
        )
        beta_ij = (
            self._beta_cov_Cxi2[:, :, np.newaxis, np.newaxis]
            * self._beta_cov_Cxi2[:, np.newaxis, :, np.newaxis]
        )

        beta_pk_ij = beta_ij * self._Pk_lambdai_lambdaj
        alpha_n_ij = alpha_ij * self._one_over_n_lambdai_lambdaj

        return alpha_n_ij, beta_pk_ij

    def _init_Cxi2_cov_arrays(self):
        # internal attributes
        #    alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be
        #    fitted on (few, ~100) simulations to correct for bias model
        #    inaccuracy, non-poissonian shot-noise and high-order terms ref
        #    values are alpha=0,beta=1,gamma=0 (see Euclid Collaboration:
        #    Fumagalli et al. 2022)
        #    cov_g, cov_ng are TWO TERMS OF EQ. 73
        self._alpha_cov_Cxi2 = np.zeros((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div))
        self._beta_cov_Cxi2 = np.ones((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div))
        self._gamma_cov_Cxi2 = np.zeros((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div))
        self._cov_g = np.zeros(
            (
                self.z_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Rad_obs_Cxi2_div,
                self.Rad_obs_Cxi2_div,
            )
        )
        self._cov_ng = np.zeros(
            (
                self.z_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div,
                self.Rad_obs_Cxi2_div,
                self.Rad_obs_Cxi2_div,
            )
        )
        # output
        self.cov_Cxi2_zbin_Lbin_Rbin = np.zeros(
            (
                self.z_obs_Cxi2_div,
                self.z_obs_Cxi2_div,
                self.Lambda_obs_Cxi2_div + 1,
                self.Lambda_obs_Cxi2_div + 1,
                self.Rad_obs_Cxi2_div,
                self.Rad_obs_Cxi2_div,
            )
        )

    def compute_Cxi2_cov(self):

        # check precomputed attributes
        if self._missing_attributes(
            self._Pk_lambdai_lambdaj,
            self._one_over_n_lambdai_lambdaj,
            self._W_rad,
            self._V_rad,
            self._V_zob,
        ):
            raise ValueError("Run compute_Cxi2 first!")

        # prepare internal attributes
        self._init_Cxi2_cov_arrays()

        # define cluster clustering bin numbers for loops
        z_bin_numbers = range(self.z_obs_Cxi2_div)
        lambda_bin_numbers = range(self.Lambda_obs_Cxi2_div)
        rad_bin_numbers = range(self.Rad_obs_Cxi2_div)

        # compute nuisance parameters
        alpha_n_ij, beta_pk_ij = self._compute_alpha_beta()

        for lambda_bin_i in lambda_bin_numbers:
            for lambda_bin_j in lambda_bin_numbers:
                for rad_bin in rad_bin_numbers:

                    self._cov_ng[
                        :,
                        lambda_bin_i,
                        lambda_bin_j,
                        lambda_bin_i,
                        lambda_bin_j,
                        rad_bin,
                        rad_bin,
                    ] = (
                        simps(
                            self.k**2.0
                            / (2.0 * np.pi**2.0)
                            * self._W_rad[:, rad_bin, :]
                            * beta_pk_ij[:, lambda_bin_i, lambda_bin_j, :],
                            x=self.k,
                        )
                        * (1 + self._gamma_cov_Cxi2[:, lambda_bin_i])
                        * self._one_over_n_lambdai_lambdaj[
                            :, lambda_bin_i, lambda_bin_i, 0
                        ]
                        * (1 + self._gamma_cov_Cxi2[:, lambda_bin_j])
                        * self._one_over_n_lambdai_lambdaj[
                            :, lambda_bin_j, lambda_bin_j, 0
                        ]
                        / self._V_rad[:, rad_bin]
                    )

                    for lambda_bin_k in lambda_bin_numbers:
                        for lambda_bin_h in lambda_bin_numbers:

                            # gaussian term
                            self._cov_g[
                                :,
                                lambda_bin_i,
                                lambda_bin_j,
                                lambda_bin_k,
                                lambda_bin_h,
                                :,
                                :,
                            ] = simps(
                                self.k**2.0
                                / (2.0 * np.pi**2.0)
                                * self._W_rad[:, np.newaxis, :, :]
                                * self._W_rad[:, :, np.newaxis, :]
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
                                x=self.k,
                                axis=-1,
                            )

        ### EQ. 89 + RESHAPE according to 2ptCF
        for z_bin in z_bin_numbers:
            for lambda_bin_i in lambda_bin_numbers:
                for lambda_bin_j in lambda_bin_numbers:
                    for lambda_bin_k in lambda_bin_numbers:
                        for lambda_bin_h in lambda_bin_numbers:
                            # NOTE: if more than 2 richness bins, this has to be modified
                            self.cov_Cxi2_zbin_Lbin_Rbin[
                                z_bin,
                                z_bin,
                                lambda_bin_i + lambda_bin_j,
                                lambda_bin_k + lambda_bin_h,
                                :,
                                :,
                            ] = (
                                1
                                / self._V_zob[z_bin]
                                * (
                                    (self._cov_g + self._cov_ng)[
                                        z_bin,
                                        lambda_bin_i,
                                        lambda_bin_j,
                                        lambda_bin_k,
                                        lambda_bin_h,
                                        :,
                                        :,
                                    ]
                                    + (self._cov_g + self._cov_ng)[
                                        z_bin,
                                        lambda_bin_i,
                                        lambda_bin_j,
                                        lambda_bin_h,
                                        lambda_bin_k,
                                        :,
                                        :,
                                    ]
                                )
                            )

    def compute_all_quantities(
        self,
        CG_like_selection: str = "CC_CWL_Cxi2",
        CG_xi2_cov_selection: str = "covCC_covCxi2",
        #        external_richness_selection_function: str = 'non_CG_ESF',
    ):
        # main function

        if CG_like_selection in ["CC", "CC_CWL", "CC_Cxi2", "CC_CWL_Cxi2"]:
            self.compute_counts()
            if CG_xi2_cov_selection in ["covCC", "covCC_covCxi2"]:
                self.compute_bias()
                self.compute_counts_cov()

        if CG_like_selection in ["CC_CWL", "CC_CWL_Cxi2"]:
            self.compute_reduced_shear(self.N_zbin_Lbin)

        ##########################################
        # 2point correlation function
        if CG_like_selection in ["CC_Cxi2", "CC_CWL_Cxi2"]:
            self.compute_Cxi2()

            # 2point correlation function covariance
            if CG_xi2_cov_selection in ["covCxi2", "covCC_covCxi2"]:
                self.compute_Cxi2_cov()
                # note: this function depends on these quantities, that are computed by the covariance function.
                # (Pk_lambdai_lambdaj one_over_n_lambdai_lambdaj,W_rad,V_rad,V_zob,)
                #   I made them class objects, but we could just put the covariance function inside the
                # cluster clustering function as I did for cluster counts


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

def weak_lensing_optical_selection_bias_correction(R, R0, A, alpha, beta, gamma):
    """
    Correction for the weak lensing optical selection bias to account for
    miscentering and projection effects. To be multiplied directly to the WL
    profile integrated in observed richness and redshift.

    Parameters
    ----------
    R: numpy.ndarray
        Radius of the profile in Mpc
    R0: numpy.ndarray
        Transition scale in Mpc, dimensions should be (z_obs_bins, lambda_obs_bins)
    A: numpy.ndarray
        Amplitude of the correction, dimensions should be (z_obs_bins, lambda_obs_bins)
    alpha: numpy.ndarray
        Slope at small radii, dimensions should be (z_obs_bins, lambda_obs_bins)
    beta: numpy.ndarray
        Slope at large radii, dimensions should be (z_obs_bins, lambda_obs_bins)
    gamma: numpy.ndarray
        Smoothness of the transition between slopes, dimensions should be (z_obs_bins, lambda_obs_bins)


    Retruns
    -------
        Correction for WL optical selection bias. Dimension (z_obs_bins, lambda_obs_bins)

    Note
    ----
    Reasonable values for the parameters are: R0=1.20cMpc/h, A=0.20, alpha=4.0,
    beta=−0.3 , gamma=1.6
    """
    return A * (R / R0) ** alpha * (1 + (R / R0**gamma)) ** ((alpha - beta) / gamma) + 1

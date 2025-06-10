Truth tables for clustering and covariance

Cosmology: <br>
$`\Omega_m`$ = 0.3158 <br>
h       = 0.6732 <br>
$`\Omega_b`$ = 0.0494 <br>
$`n_s`$     = 0.9661 <br>
$`\log A_s`$ = -8.679490
$'m_\nu'$   = 0.06

Eq 19: <br>
$`A_\lambda`$ = 52.0 <br>
$`B_\lambda`$ = 0.9 <br>
$`C_\lambda`$ = 0.5


Eq 20: <br>
$`A_\sigma`$ = 0.2 <br>
$`B_\sigma`$ = -0.05 <br>
$`C_\sigma`$ = 0.001


area: <br>
$`\theta = \pi`$/3 <br>
$`\Omega_{\rm sky}`$ = 10313 deg$`^2`$


Redshift bins <br>
z_bins = np.arange(0.2, 1.81, 0.4)

Radial bins <br>
r_bins = np.geomspace(20,130,31)  (Mpc/h)

Richness bins <br>
l_bins = np.array([20,30,500])


Halo mass function: Castro22
Halo bias: Castro23

No IR resummation

file: xi_CG_CL_xi_and_cov.npz
Contains xi(z,lambda,r), cov(z,z,lambda,lambda,r,r)

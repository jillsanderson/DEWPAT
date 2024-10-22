import numpy as np
import scipy.linalg as SLA

def gaussian_prob_divergence(mode, means, covs):
    """
    Computes a distributional divergence between moments based on an underlying 
        Gaussian prior assumption.
    This is of course equivalent to the Bayesian maximum entropy principle, 
        so that minimal information (least structure) is built into the prior 
        (with the prescribed moments).

    Precomputes the necessary information, and then delegates to the method itself.

    TODO: should support multiple calls so we can share precomputations over modes
    """
    # Fixed definitions
    metrics = [ 'pw_symmetric_KL', 'pw_W2', 'pw_Hellinger',
                'pw_bhattacharyya', 'pw_FMATF' ]
    distinfo = { 'pw_symmetric_KL'  : [ 'inverse'  ], 
                 'pw_W2'            : [ 'mat_sqrt' ], 
                 'pw_Hellinger'     : [ 'det_C'    ],
                 'pw_bhattacharyya' : [ 'det_C'    ],
                 'pw_FMATF'         : [ 'inverse'  ], }
    # Checks
    assert mode in metrics
    assert len(means) == len(covs)
    for _mu, _sigma in zip(means, covs):
        _mm = np.ma.is_masked(_mu)
        _sm = np.ma.is_masked(_sigma)
        assert not _mm and not _sm, str(_mm) + ", " + str(_sm)

    ### Precomputations ###
    pre_reqs = distinfo[mode]
    eps = 1e-7
    I_eps = np.eye(means[0].shape[0]) * eps

    # Remove masks
    means = [ mean.data for mean in means ]
    covs  = [ cov.data for cov in covs ]

    # Covariance Inverses
    inverses = None
    if 'inverse' in pre_reqs:
        inverses = [ np.linalg.inv(C + I_eps) for C in covs ]

    # Covariance Matrix Square Roots
    C_sqrts = None
    if 'mat_sqrt' in pre_reqs:
        C_sqrts = [ SLA.sqrtm(C + I_eps) for C in covs ]

    # Covariance determinants
    det_Cs = None
    if 'det_C' in pre_reqs:
        det_Cs = [ np.linalg.det(C + I_eps) for C in covs ]

    ### Distance Computations ###
    n = len(means)
    D = -np.ones( (n, n) )
    # Assume the metric is symmetric
    for i in range(n):
        for j in range(i, n):
            mu1 = means[i]
            mu2 = means[j]
            C1  = covs[i]
            C2  = covs[j]
            # Divergence computation
            if   mode == 'pw_symmetric_KL':
                D[i, j] = jeffreys_div_gauss(mu1, mu2, C1, C2, inverses[i], inverses[j])
            elif mode == 'pw_W2':
                D[i, j] = wass2_div_gauss(mu1, mu2, C1, C2, C_sqrts[i], C_sqrts[j])
            elif mode == 'pw_Hellinger':
                D[i, j] = hellinger_div_gauss(mu1, mu2, C1, C2, det_Cs[i], det_Cs[j])
            elif mode == 'pw_bhattacharyya':
                D[i, j] = bhattacharyya_div_gauss(mu1, mu2, C1, C2, det_Cs[i], det_Cs[j])
            elif mode == 'pw_FMATF':
                D[i, j] = FMAF_div_gauss(mu1, mu2, C1, C2, sigma2_inverse = inverses[j])
            # Symmetric insertion
            D[j, i] = D[i, j]
    return D

### Individual Gaussian divergences ###

def jeffreys_div_gauss(mu1, mu2, sigma1, sigma2, sigma1_inverse, sigma2_inverse):
    """ J-divergence (Symmetrized KL divergence) between two Gaussians """
    I = np.eye(mu1.shape[0])
    C_A = 0.5 * (sigma1 + sigma2)
    mean_metric = C_A_inv_scaled_mean_dist(mu1, mu2, C_A + I*1e-7)
    cov_term = 0.5 * np.trace( sigma1_inverse @ sigma2 + sigma2_inverse @ sigma1 - 2*I )
    return mean_metric + cov_term

def wass2_div_gauss(mu1, mu2, sigma1, sigma2, sigma1_sqrt, sigma2_sqrt):
    """ Wasserstein-2 divergence between two Gaussians """
    mean_diff = ( (mu1 - mu2)**2 ).sum()
    I = np.eye(mu1.shape[0])
    inner_bal = SLA.sqrtm( sigma2_sqrt @ sigma1 @ sigma2_sqrt + I*1e-7 ) 
    cov_diff  = np.trace( sigma1 + sigma2 - 2*inner_bal)
    return np.clip(mean_diff + cov_diff, a_min = 1e-7, a_max = None)

def hellinger_div_gauss(mu1, mu2, sigma1, sigma2, det_sigma1, det_sigma2):
    """ Hellinger divergence between two Gaussians (technically the squared Hellinger distance) """
    C_A = 0.5 * (sigma1 + sigma2) + np.eye(mu1.shape[0])*1e-7
    coef = np.power(det_sigma1, 0.25) * np.power(det_sigma2, 0.25) / np.sqrt( np.linalg.det(C_A) + 1e-7 )
    exp_term = np.exp( -1.0 * C_A_inv_scaled_mean_dist(mu1, mu2, C_A) / 8.0 )
    return 1.0 - coef * exp_term

def bhattacharyya_div_gauss(mu1, mu2, sigma1, sigma2, det_sigma1, det_sigma2):
    """ Bhattacharyya distance between two Gaussians """
    C_A = 0.5 * (sigma1 + sigma2) + np.eye(mu1.shape[0])*1e-7
    det_C_A = np.linalg.det(C_A)
    mean_diff = C_A_inv_scaled_mean_dist(mu1, mu2, C_A) / 8.0
    cov_term = 0.5 * np.log( det_C_A / np.sqrt( det_sigma1 * det_sigma2 + 1e-7 ) )
    return mean_diff + cov_term

def FMAF_div_gauss(mu1, mu2, sigma1, sigma2, sigma2_inverse, eps=1e-6):
    """ The Abou-Mustafa, Torres, and Ferries divergence using the Forstner-Moonen Covariance metric """ 
    C_A = 0.5 * (sigma1 + sigma2) + np.eye(mu1.shape[0])*eps
    #print('ca', C_A)
    mean_diff = np.sqrt( C_A_inv_scaled_mean_dist(mu1, mu2, C_A) + eps )
    #print('md', mean_diff)
    # lambda = eigenvalues(sigma_1 sigma_2_inverse), so sigma_1 V = lambda sigma_2 V
    B = (sigma1 + np.eye(mu1.shape[0])*eps) @ sigma2_inverse
    #print('B', B)
    lambda_evals = np.absolute( np.linalg.eigvals( B ) )
    #print(lambda_evals, 'lam')
    cov_term = np.sqrt( (np.log(lambda_evals + eps)**2).sum() + eps )
    #print('ct', cov_term)
    return mean_diff + cov_term

### Helper functions ###

def C_A_inv_scaled_mean_dist(mu1, mu2, C_A):
    delta = mu2 - mu1
    return delta @ np.linalg.inv(C_A) @ delta



#

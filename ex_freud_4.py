import numpy as np
from UncertainSCI.ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC, hankel_deter, mod_cheb, dPI4
from UncertainSCI.utils.compute_moment import compute_freud_moment
from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials
import time
from tqdm import tqdm
import scipy.io
ab_true = scipy.io.loadmat('ab_freud_4.mat')['coeff']
t_vpa = scipy.io.loadmat('time_freud_4.mat')['time']

"""
We use six methods

1. pc (Predictor-corrector method)
2. sp (Stieltjes procedure)
3. apc (Arbitrary polynomial chaos expansion method)
4. hd (Hankel determinants)
5. mc (Modified Chebyshev algorithm)
6. dp (Discrete Painlevé I equation method)

to compute the recurrence coefficients for
the freud weight function when alpha = 4.
"""

a = -np.inf
b = np.inf


def weight(x):
    return np.exp(-x**4)


singularity_list = []

N_array = [20, 40, 60, 80, 100]

t_pc = np.zeros(len(N_array))
t_sp = np.zeros(len(N_array))
t_apc = np.zeros(len(N_array))
t_hd = np.zeros(len(N_array))
t_mc = np.zeros(len(N_array))
t_dp = np.zeros(len(N_array))

e_pc = np.zeros(len(N_array))
e_sp = np.zeros(len(N_array))
e_apc = np.zeros(len(N_array))
e_hd = np.zeros(len(N_array))
e_mc = np.zeros(len(N_array))
e_dp = np.zeros(len(N_array))

iter_n = np.arange(100)
for k in tqdm(iter_n):
    for ind, N in enumerate(N_array):
        ab = ab_true[:N]

        m = compute_freud_moment(rho=0, m=4, n=N)

        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(ab - ab_pc, None)

        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_unbounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_sp[ind] += (end - start) / len(iter_n)
        e_sp[ind] = np.linalg.norm(ab - ab_sp, None)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_apc = aPC(m, N)
        end = time.time()
        t_apc[ind] += (end - start) / len(iter_n)
        e_apc[ind] = np.linalg.norm(ab - ab_apc, None)

        # Hankel Determinant
        start = time.time()
        ab_hd = hankel_deter(N, m)
        end = time.time()
        t_hd[ind] += (end - start) / len(iter_n)
        e_hd[ind] = np.linalg.norm(ab - ab_hd, None)

        # Modified Chebyshev
        H = HermitePolynomials(probability_measure=False)

        def peval(x, n):
            return H.eval(x, n)

        def integrand(x):
            return weight(x) * peval(x, i).flatten()
        mod_m = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            mod_m[i] = gq_modification_unbounded_composite(integrand,
                                                           a, b, 10,
                                                           singularity_list)
        start = time.time()
        ab_mc = mod_cheb(N, mod_m, H)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(ab - ab_mc, None)

        # Discrete Painleve Equation I
        start = time.time()
        ab_dp = dPI4(N)
        end = time.time()
        t_dp[ind] += (end - start) / len(iter_n)
        e_dp[ind] = np.linalg.norm(ab - ab_dp, None)


"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

--- Frobenius norm error ---

e_pc
array([3.10475162e-15, 5.38559426e-15, 9.24185904e-15,
1.11959041e-14, 1.33658569e-14])

e_sp
array([8.91113940e-15, 1.42689811e-14, 3.02075444e-14,
5.76568141e-14, 8.63063396e-14])

e_apc
array([3.97816821e-08, nan, nan, nan, nan])

e_hd
array([5.96070448e-08, nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan, nan])

e_dp
array([4.4963574e-07, nan, nan, nan, nan])


--- elapsed time ---

t_vpa
array([18.85687984, 99.38145614, 293.44349978, 631.20487295, 1196.28614909])

t_pc
array([0.24657346, 0.74928286, 1.59986169, 2.71804272, 4.12271183])

t_sp
array([0.23871661, 0.74802218, 1.59658725, 2.71879307, 4.12151607])

t_apc
array([0.00131583, 0.00626945, 0.00826812, 0.01558006, 0.02475297])

t_hd
array([0.00270291, 0.00937102, 0.01885587, 0.03280065, 0.05145703])

t_mc
array([0.00151602, 0.00616114, 0.01425739, 0.0256006 , 0.04030894])

t_dp
array([4.32443619e-05, 7.47990608e-05, 1.02679729e-04, 1.38518810e-04,
1.60083771e-04])

"""

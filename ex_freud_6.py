import numpy as np
from UncertainSCI.ttr import predict_correct_unbounded, stieltjes_unbounded, \
        aPC, hankel_deter, mod_cheb, dPI6
from UncertainSCI.utils.compute_moment import compute_freud_moment
from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials
import time
from tqdm import tqdm
import scipy.io
ab_true = scipy.io.loadmat('ab_freud_6.mat')['coeff']
t_vpa = scipy.io.loadmat('time_freud_6.mat')['time']

"""
We use six methods

1. pc (Predictor-corrector method)
2. sp (Stieltjes procedure)
3. apc (Arbitrary polynomial chaos expansion method)
4. hd (Hankel determinants)
5. mc (Modified Chebyshev algorithm)
6. dp (Discrete Painlevé I equation method)

to compute the recurrence coefficients for
the freud weight function when m = 6.
"""

a = -np.inf
b = np.inf


def weight(x):
    return np.exp(-x**6)


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

        m = compute_freud_moment(rho=0, m=6, n=N)

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
        ab_dp = dPI6(N)
        end = time.time()
        t_dp[ind] += (end - start) / len(iter_n)
        e_dp[ind] = np.linalg.norm(ab - ab_dp, None)


"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

--- Frobenius norm error ---

e_pc
array([2.59281844e-14, 2.64537346e-14, 2.68816085e-14, 2.75287150e-14,
2.81177506e-14])

e_sp
array([1.67486852e-14, 3.94620768e-14, 5.85110601e-14, 1.08525933e-13,
1.37666177e-13])

e_apc
array([2.07534802e-06, nan, nan, nan, nan])

e_hd
array([1.6690485e-06, nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan, nan])

e_dp
array([3.37781893e-07, nan, nan, nan, nan])


--- elapsed time ---

t_vpa
array([19.13496825, 101.10079443, 300.40989682, 633.19571353, 1362.86210165])

t_pc
array([0.22065904, 0.6508228 , 1.33960142, 2.27026976, 3.40105817])

t_sp
array([0.00152493, 0.00414109, 0.00819707, 0.01298094, 0.02004313])

t_apc
array([0.00123407, 0.00366557, 0.00751336, 0.01272842, 0.01981904])

t_hd
array([0.00280349, 0.00933857, 0.01942751, 0.0347622 , 0.05435012])

t_mc
array([0.0015401 , 0.00630336, 0.01466616, 0.02608238, 0.04101465])

t_dp
array([0.00012545, 0.00024034, 0.00036499, 0.00048342, 0.0005873 ])

"""

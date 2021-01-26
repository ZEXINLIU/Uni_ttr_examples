import numpy as np
from UncertainSCI.ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC, hankel_deter, mod_cheb, lanczos_stable
from UncertainSCI.utils.compute_moment import compute_moment_discrete
from UncertainSCI.families import JacobiPolynomials
import time
from tqdm import tqdm

"""
We use six methods

1. pc (Predictor-corrector method)
2. sp (Stieltjes procedure)
3. apc (Arbitrary polynomial chaos expansion method)
4. hd (Hankel determinants)
5. mc (Modified Chebyshev algorithm)
6. lz (Stabilized Lanczos algorithm)

to compute the recurrence coefficients for
the discrete Chebyshev transformed to [0,1).
"""


def discrete_chebyshev(N):
    """
    Return the first N exact recurrence coefficients
    """
    ab = np.zeros([N, 2])
    ab[1:, 0] = (N-1) / (2*N)
    ab[0, 1] = 1.
    ab[1:, 1] = np.sqrt(1/4 * (1 - (np.arange(1, N)/N)**2)
                        / (4 - (1/np.arange(1, N)**2)))
    return ab


# N_array = [37, 38, 39, 40]
# N_quad = 40
# N_array = [56, 60, 64, 68]
# N_quad = 80
# N_array = [82, 89, 96, 103]
# N_quad = 160
N_array = [82, 89, 96, 103]
N_quad = 320

x = np.arange(N_quad) / N_quad
w = (1/N_quad) * np.ones(len(x))

t_pc = np.zeros(len(N_array))
t_sp = np.zeros(len(N_array))
t_apc = np.zeros(len(N_array))
t_hd = np.zeros(len(N_array))
t_mc = np.zeros(len(N_array))
t_lz = np.zeros(len(N_array))

e_pc = np.zeros(len(N_array))
e_sp = np.zeros(len(N_array))
e_apc = np.zeros(len(N_array))
e_hd = np.zeros(len(N_array))
e_mc = np.zeros(len(N_array))
e_lz = np.zeros(len(N_array))

iter_n = np.arange(100)
for k in tqdm(iter_n):
    for ind, N in enumerate(N_array):
        ab = discrete_chebyshev(N_quad)[:N, :]

        m = compute_moment_discrete(x, w, N)

        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_discrete(x, w, N)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(ab - ab_pc, None)

        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_discrete(x, w, N)
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
        J = JacobiPolynomials(probability_measure=False)

        def peval(x, n):
            return J.eval(x, n)

        def integrand(x):
            return peval(x, i).flatten()
        mod_m = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            mod_m[i] = np.sum(integrand(x) * w)
        start = time.time()
        ab_mc = mod_cheb(N, mod_m, J)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(ab - ab_mc, None)

        # Stabilized Lanczos
        start = time.time()
        ab_lz = lanczos_stable(x, w, N)
        end = time.time()
        t_lz[ind] += (end - start) / len(iter_n)
        e_lz[ind] += np.linalg.norm(ab - ab_lz, None)


"""
N_array = [37, 38, 39, 40] with tol = 1e-12, N_quad = 40

--- Frobenius norm error ---

e_pc
array([5.83032276e-16, 7.88106850e-16, 1.31264360e-14, 6.81247807e-13])

e_sp
array([6.79107529e-15, 7.08424027e-15, 1.52208335e-14, 7.23359604e-13])

e_apc
array([nan, nan, nan, nan])

e_hd
array([nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan])

e_lz
array([8.26282134e-16, 8.75621328e-16, 8.78366402e-16, 8.80556299e-16])

--- elapsed time ---

t_pc
array([0.01866756, 0.01940269, 0.02026843, 0.02117965])

t_sp
array([0.01808646, 0.01872314, 0.01958155, 0.02055171])

t_apc
array([0.00344686, 0.00372854, 0.00387698, 0.00402875])

t_hd
array([0.00818913, 0.00850275, 0.00893114, 0.00921517])

t_mc
array([0.00544071, 0.00575021, 0.00612659, 0.00639981])

t_lz
array([0.00161063, 0.00168495, 0.00170782, 0.00174096])



N_array = [56, 60, 64, 68] with tol = 1e-12, N_quad = 80

e_pc
array([1.19606888e-15, 1.92721740e-13, 5.03366337e-10, 3.84167092e-06])

e_sp
array([3.81010361e-15, 7.60074466e-14, 2.02231318e-10, 1.57318802e-06])

e_apc
array([nan, nan, nan, nan, nan])

e_hd
array([nan, nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan, nan])

e_lz
array([1.15977130e-15, 1.21238184e-15, 1.36341761e-15, 1.49468349e-15])


t_pc
array([0.04124258, 0.0486698 , 0.05391277, 0.05956687])

t_sp
array([0.04043174, 0.04731631, 0.05250208, 0.05827137])

t_apc
array([0.00683582, 0.00755854, 0.00840556, 0.00946519])

t_hd
array([0.01683453, 0.01991775, 0.02230049, 0.02437497])

t_mc
array([0.01336397, 0.01488232, 0.01709907, 0.01894911])

t_lz
array([0.0028906 , 0.00300488, 0.00327993, 0.00346822])



N_array = [82, 89, 96, 103] with tol = 1e-12, N_quad = 160

e_pc
array([1.35320885e-15, 1.52422750e-12, 1.12490901e-08, 2.16713303e-04])

e_sp
array([6.44431630e-15, 3.66258846e-12, 2.71222200e-08, 5.23466153e-04])

e_apc
array([nan, nan, nan, nan])

e_hd
array([nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan])

e_lz
array([1.32966300e-15, 1.41362828e-15, 1.55629351e-15, 1.68556574e-15])


t_pc
array([0.10012377, 0.11433365, 0.13067236, 0.15082069])

t_sp
array([0.09506917, 0.11128752, 0.12852232, 0.1470592 ])

t_apc
array([0.01341118, 0.01552454, 0.01833375, 0.02090821])

t_hd
array([0.03509946, 0.04140449, 0.04904011, 0.05577155])

t_mc
array([0.02791258, 0.03276293, 0.03802878, 0.04396228])

t_lz
array([0.00592635, 0.00665268, 0.00714997, 0.00809739])



N_array = [82, 89, 96, 103] with tol = 1e-12, N_quad = 320

e_pc
array([1.19348975e-15, 1.33976368e-15, 1.57963123e-15, 1.73577787e-15])

e_sp
array([2.92199121e-15, 3.03780940e-15, 3.42385023e-15, 3.63905129e-15])

e_apc
array([nan, nan, nan, nan])

e_hd
array([nan, nan, nan, nan])

e_mc
array([nan, nan, nan, nan])

e_lz
array([1.18636824e-15, 1.35263944e-15, 1.65349634e-15, 1.79683860e-15])


t_pc
array([0.12287572, 0.13825425, 0.16237012, 0.18260074])

t_sp
array([0.11560148, 0.13418031, 0.15452703, 0.17811085])

t_apc
array([0.01396315, 0.01658385, 0.01925649, 0.02249643])

t_hd
array([0.03557385, 0.04164304, 0.04904677, 0.05764251])

t_mc
array([0.02806302, 0.03326251, 0.03876049, 0.04441474])

t_lz
array([0.01207455, 0.01389778, 0.0154752 , 0.01657487])

"""

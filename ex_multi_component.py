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
the Chebyshev weight function plus a discrete measure.

Here we test for with one mass point at t = −1 with strength y = 0.5
can extend to multiple masses inside [-1,1], e.x. 3 masses --- t = [-1,1,0]
See Example 2.39 in Gautschi's book

case of one mass at t = 2, outside [-1,1] with strength y = 1
Stieltjes’s procedure becomes extremely unstable
if one or more mass points are located outside [−1, 1]
Lanczos’s algorithm is imperative
"""

alpha = -0.6
beta = 0.4

internal = False
if internal:
    t = np.array([-1.])
    y = np.array([0.5])
    ab = np.array([[3.7037037037e-2, 1.5000000000e0],
                   [3.2391629514e-2, 2.3060042904e-1],
                   [4.4564744879e-3, 2.4754733005e-1],
                   [8.6966173737e-4, 2.4953594220e-1]])
else:
    t = np.array([2.])
    y = np.array([1.])
    ab = np.array([[1.2777777778e0, 2.0000000000e0],
                   [-1.9575723334e-3, 2.4959807576e-1],
                   [-1.9175655273e-4, 2.4998241443e-1],
                   [-3.4316341540e-5, 2.4999770643e-1]])

ab[:, 1] = np.sqrt(ab[:, 1])

N_array = [1, 7, 18, 40]

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
        """
        assume the first N coefficients are required, {a_n,b_n}_{n=0}^{N-1},
        then the maximum degree occuring in Stieltjes is 2(N-1)+1 = 2N-1,
        i.e. int q(x) dmu = sum_{j=1}^(?) w_j q(x_j) is exact
        (converge after the very first iteration step, N_0^[1] = N_0^[0]+1)
        for N_0^[0] = 1 + np.floor((2N-1)/2) when Gauss-type rules
        for N_0^[0] = 1 + np.floor((2N-1)/1) when interpolatory rules
        """
        # N_quad = int(1 + np.floor((2*N-1)/2)+1) # N_0^[1] = N_0^[0]+1)
        N_quad = N
        xg, wg = JacobiPolynomials(alpha, beta).gauss_quadrature(N_quad)
        xg = np.hstack([xg, t])
        wg = np.hstack([wg, y])

        m = compute_moment_discrete(xg, wg, N+1)

        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_discrete(xg, wg, N+1)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(np.array([ab_pc[N, 0], ab_pc[N-1, 1]])
                                   - ab[ind])

        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_discrete(xg, wg, N+1)
        end = time.time()
        t_sp[ind] += (end - start) / len(iter_n)
        e_sp[ind] = np.linalg.norm(np.array([ab_sp[N, 0], ab_sp[N-1, 1]])
                                   - ab[ind])

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_apc = aPC(m, N+1)
        end = time.time()
        t_apc[ind] += (end - start) / len(iter_n)
        e_apc[ind] = np.linalg.norm(np.array([ab_apc[N, 0], ab_apc[N-1, 1]])
                                    - ab[ind])

        # Hankel Determinant
        start = time.time()
        ab_hd = hankel_deter(N+1, m)
        end = time.time()
        t_hd[ind] += (end - start) / len(iter_n)
        e_hd[ind] = np.linalg.norm(np.array([ab_hd[N, 0], ab_hd[N-1, 1]])
                                   - ab[ind])

        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)

        def peval(x, n):
            return J.eval(x, n)

        def integrand(x):
            return peval(x, i).flatten()
        mod_m = np.zeros(2*(N+1) - 1)
        for i in range(2*(N+1) - 1):
            mod_m[i] = np.sum(integrand(xg) * wg)
        start = time.time()
        ab_mc = mod_cheb(N+1, mod_m, J)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(np.array([ab_mc[N, 0], ab_mc[N-1, 1]])
                                   - ab[ind])

        # Stabilized Lanczos
        start = time.time()
        ab_lz = lanczos_stable(xg, wg, N+1)
        end = time.time()
        t_lz[ind] += (end - start) / len(iter_n)
        e_lz[ind] = np.linalg.norm(np.array([ab_lz[N, 0], ab_lz[N-1, 1]])
                                   - ab[ind])


"""
N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at the left end point −1, inside [-1,1]

--- Frobenius norm error ---
e_pc
array([3.70536934e-14, 3.63411982e-12, 3.02486054e-12, 3.90315660e-12])

e_sp
array([3.70883879e-14, 3.63411663e-12, 3.02502766e-12, 3.90315700e-12])

e_apc
array([3.70536934e-14, 3.54462473e-12, 1.67021818e-04,            nan])

e_hd
array([3.70536934e-14, 3.63564139e-12, 1.71559866e-04,            nan])

e_mc
array([3.70675712e-14, 3.63365277e-12, 3.02025653e-12, 3.86585472e-12])

e_lz
array([3.70328768e-14, 3.63428905e-12, 3.02508255e-12, 3.90321203e-12])

--- elapsed time ---


N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at 2, outside [-1,1]

e_pc
array([2.22224461e-11, 5.43911845e-13, 3.80059322e-12, 2.48812425e-06])

e_sp
array([2.22224461e-11, 5.44326447e-13, 3.80245850e-12, 2.48245648e-06])

e_apc
array([2.22222241e-11, 1.80879211e-09,            nan,            nan])

e_hd
array([2.22222241e-11, 1.80880344e-09,            nan,            nan])

e_mc
array([2.22222241e-11, 8.90316971e-11, 2.90453847e+00, 1.84279278e+00])

e_lz
array([2.22224461e-11, 5.44072730e-13, 3.80229185e-12, 2.10192975e-12])

--- elapsed time ---

"""

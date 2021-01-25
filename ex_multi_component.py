import numpy as np

from UncertainSCI.ttr.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC, hankel_deter, mod_cheb, lanczos_stable

from UncertainSCI.utils.compute_moment import compute_moment_discrete
from UncertainSCI.families import JacobiPolynomials

import time
from tqdm import tqdm

import pdb
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


t = np.array([-1.])
y = np.array([0.5])
ab = np.array([[3.7037037037e-2, 1.5000000000e0], \
        [3.2391629514e-2, 2.3060042904e-1], \
        [4.4564744879e-3, 2.4754733005e-1], \
        [8.6966173737e-4, 2.4953594220e-1]])

# t = np.array([2.])
# y = np.array([1.])
# ab = np.array([[1.2777777778e0, 2.0000000000e0], \
        # [-1.9575723334e-3, 2.4959807576e-1], \
        # [-1.9175655273e-4, 2.4998241443e-1], \
        # [-3.4316341540e-5, 2.4999770643e-1]])


ab[:,1] = np.sqrt(ab[:,1])

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
        i.e. \int q(x) d\mu = \sum_{j=1}^(?) w_j q(x_j) is exact
        (converge after the very first iteration step, N_0^[1] = N_0^[0]+1)
        for N_0^[0] = 1 + np.floor((2N-1)/2) when Gauss-type rules
        for N_0^[0] = 1 + np.floor((2N-1)/1) when interpolatory rules

        """
        # N_quad = int(1 + np.floor((2*N-1)/2)+1) # N_0^[1] = N_0^[0]+1)
        N_quad = N
        xg,wg = JacobiPolynomials(alpha, beta).gauss_quadrature(N_quad)
        xg = np.hstack([xg, t]); wg = np.hstack([wg, y])

        m = compute_moment_discrete(xg, wg, N+1)

        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_discrete(xg, wg, N+1)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(np.array([ab_pc[N,0], ab_pc[N-1,1]]) - ab[ind])


        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_discrete(xg, wg, N+1)
        end = time.time()
        t_sp[ind] += (end - start) / len(iter_n)
        e_sp[ind] = np.linalg.norm(np.array([ab_sp[N,0], ab_sp[N-1,1]]) - ab[ind])
        

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_apc = aPC(m, N+1)
        end = time.time()
        t_apc[ind] += (end - start) / len(iter_n)
        e_apc[ind] = np.linalg.norm(np.array([ab_apc[N,0], ab_apc[N-1,1]]) - ab[ind])
        
        
        # Hankel Determinant
        start = time.time()
        ab_hd = hankel_deter(N+1, m)
        end = time.time()
        t_hd[ind] += (end - start) / len(iter_n)
        e_hd[ind] = np.linalg.norm(np.array([ab_hd[N,0], ab_hd[N-1,1]]) - ab[ind])
        
    
        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_m = np.zeros(2*(N+1) - 1)
        for i in range(2*(N+1) - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_m[i] = np.sum(integrand(xg) * wg)
        start = time.time()
        ab_mc = mod_cheb(N+1, mod_m, J)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(np.array([ab_mc[N,0], ab_mc[N-1,1]]) - ab[ind])


        # Stabilized Lanczos
        start = time.time()
        ab_lz = lanczos_stable(xg, wg, N+1)
        end = time.time()
        t_lz[ind] += (end - start) / len(iter_n)
        e_lz[ind] = np.linalg.norm(np.array([ab_lz[N,0], ab_lz[N-1,1]]) - ab[ind])


"""
N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at the left end point −1, inside [-1,1]

--- Frobenius norm error ---

e_pc
array([3.62931322e-14, 3.63379416e-12, 3.02480572e-12, 3.90315660e-12])

e_sp
array([3.63139473e-14, 3.63371281e-12, 3.02469426e-12, 3.90315700e-12])

e_apc
array([3.62792554e-14, 3.48944329e-12, 4.03050567e-05,            nan])

e_hd
array([3.62723171e-14, 3.48935880e-12, 3.19515452e-05,            nan])

e_mc
array([3.63139473e-14, 3.63379736e-12, 3.02308235e-12, 3.86585472e-12])

e_lz
array([3.62306868e-14, 3.63368880e-12, 3.02474993e-12, 3.90321203e-12]) 

--- elapsed time ---

t_pc
array([0.00021199, 0.0012925 , 0.00555118, 0.02309394])

t_sp
array([0.00010542, 0.00110081, 0.00520177, 0.02193147])

t_apc
array([0.00011349, 0.00040475, 0.00121665, 0.00403611])

t_hd
array([5.35368919e-05, 7.06861019e-04, 2.51838207e-03, 9.79724407e-03])

t_mc
array([8.21280479e-05, 2.61638165e-04, 1.42495394e-03, 6.64503813e-03])

t_lz
array([0.00015115, 0.00039568, 0.00097868, 0.0018136 ])


N_array = [1, 7, 18, 40] with tol = 1e-12, M = 40,
one mass at 2, outside [-1,1]

e_pc
array([2.22220020e-11, 5.44056336e-13, 3.80219138e-12, 2.48812425e-06])

e_sp
array([2.22217800e-11, 5.44016049e-13, 3.80334655e-12, 2.48245648e-06])

e_apc
array([2.22217800e-11, 4.82314808e-10,            nan,            nan])

e_hd
array([2.22217800e-11, 4.81568826e-10,            nan,            nan])

e_mc
array([2.22220020e-11, 3.51137206e-10, 3.24667579e+00, 1.84279278e+00])

e_lz
array([2.22217800e-11, 5.44064504e-13, 3.80340218e-12, 2.10192975e-12])

--- elapsed time ---

t_pc
array([0.00024996, 0.00126181, 0.00577379, 0.02348736])

t_sp
array([0.00011142, 0.00106978, 0.00525815, 0.02240052])

t_apc
array([0.00011971, 0.00040447, 0.00124131, 0.0041701 ])

t_hd
array([3.51142883e-05, 7.38089085e-04, 2.64613390e-03, 9.90080833e-03])

t_mc
array([7.98487663e-05, 2.68192291e-04, 1.46774769e-03, 6.76751137e-03])

t_lz
array([0.00014665, 0.00041877, 0.00103013, 0.00189208])

"""

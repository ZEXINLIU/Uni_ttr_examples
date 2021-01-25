import numpy as np

from UncertainSCI.ttr.compute_ttr import predict_correct_bounded, stieltjes_bounded, \
        aPC, hankel_deter, mod_cheb

from UncertainSCI.utils.compute_moment import compute_moment_bounded

from UncertainSCI.utils.compute_subintervals import compute_subintervals
from UncertainSCI.utils.quad import gq_modification_composite
from UncertainSCI.families import JacobiPolynomials


import scipy.integrate as integrate
import scipy.special as sp

import time
from tqdm import tqdm

"""
We use five methods

1. pc (Predictor-corrector method)
2. sp (Stieltjes procedure)
3. apc (Arbitrary polynomial chaos expansion method)
4. hd (Hankel determinants)
5. mc (Modified Chebyshev algorithm)

to compute the recurrence coefficients for the piecewise weight function.
"""

a = -1.
b = 1.

xi = 1/10
yita = (1-xi)/(1+xi)
gm = 1
p = -1/2
q = -1/2

def ab_pws1(N):
    """
    gm = 1, p = q = -1/2
    """
    
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = np.pi
    if N == 0:
        return ab
    b[1] = 1/2 * (1+xi**2)
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2 * (1+yita**(2*i-2)) / (1+yita**(2*i))
        b[2*i+1] = 1/4 * (1+xi)**2 * (1+yita**(2*i+2)) / (1+yita**(2*i))
    return np.sqrt(ab[:N+1,:])

def ab_pws2(N):
    """
    gm = -1, p = q = -1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = np.pi/xi
    if N == 0:
        return ab
    b[1] = xi
    if N == 1:
        return ab
    b[2] = 1/2 * (1-xi)**2
    if N == 2:
        return ab
    for i in range(1, N):
        b[2*i+1] = 1/4 * (1+xi)**2
    for i in range(2, N):
        b[2*i] = 1/4 * (1-xi)**2
    return np.sqrt(ab[:N+1,:])

def ab_pws3(N):
    """
    gm = 1, p = q = 1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    b[0] = (1-xi**2)**2 * sp.gamma(3/2) * sp.gamma(3/2) / sp.gamma(3)
    if N == 0:
        return ab
    b[1] = 1/4 * (1+xi)**2 * (1-yita**(2*0+4)) / (1-yita**(2*0+2))
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2 * (1-yita**(2*i)) / (1-yita**(2*i+2))
        b[2*i+1] = 1/4 * (1+xi)**2 * (1-yita**(2*i+4)) / (1-yita**(2*i+2))
    return np.sqrt(ab[:N+1,:])

def ab_pws4(N):
    """
    gm = -1, p = q = 1/2
    """
    ab = np.zeros((2*N,2))
    b = ab[:,1]
    
    z = -(1+xi**2)/(1-xi**2)
    F = integrate.quad(lambda x: (1-x**2)**(1/2) * (x-z)**(-1), -1, 1)[0]
    b[0] = 1/2 * (1-xi**2) * F
    if N == 0:
        return ab
    b[1] = 1/4 * (1+xi)**2
    if N == 1:
        return ab
    for i in range(1, N):
        b[2*i] = 1/4 * (1-xi)**2
        b[2*i+1] = 1/4 * (1+xi)**2
    return np.sqrt(ab[:N+1,:])

def weight(x):
    return np.piecewise(x, [np.abs(x)<xi, np.abs(x)>=xi], \
            [lambda x: np.zeros(x.size), \
            lambda x: np.abs(x)**gm * (x**2-xi**2)**p * (1-x**2)**q])

singularity_list = [ [-1, 0, q],
                     [-xi, p, 0],
                     [xi, 0, p],
                     [1, q, 0]
                     ]

N_array = [20, 40, 60, 80, 100]

t_pc = np.zeros(len(N_array))
t_sp = np.zeros(len(N_array))
t_apc = np.zeros(len(N_array))
t_hd = np.zeros(len(N_array))
t_mc = np.zeros(len(N_array))

e_pc = np.zeros(len(N_array))
e_sp = np.zeros(len(N_array))
e_apc = np.zeros(len(N_array))
e_hd = np.zeros(len(N_array))
e_mc = np.zeros(len(N_array))

iter_n = np.arange(10)
for k in tqdm(iter_n):
    
    for ind, N in enumerate(N_array):
        
        ab = ab_pws1(N)[:N]

        m = compute_moment_bounded(a, b, weight, N, singularity_list)

        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_bounded(a, b, weight, N, singularity_list)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(ab - ab_pc, None)

        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_bounded(a, b, weight, N, singularity_list)
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
        peval = lambda x, n: J.eval(x, n)
        subintervals = compute_subintervals(a, b, singularity_list)
        mod_m = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: weight(x) * peval(x,i).flatten()
            mod_m[i] = gq_modification_composite(integrand, a, b, 10, subintervals)
        start = time.time()
        ab_mc = mod_cheb(N, mod_m, J)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(ab - ab_mc, None)


"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

case pws1 (gm = 1, p = q = -1/2)

--- Frobenius norm error ---

e_pc
array([9.07801721e-15, 1.80114070e-14, 3.12765607e-14, 5.14165169e-14, 7.27067791e-14])

e_sp
array([4.73425186e-14, 2.85017480e-13, 3.85242226e-13, 3.99304271e-13, 4.62482224e-13])

e_aPC
array([0.06045599,        nan,        nan,        nan,        nan])

e_hd
array([0.06046302, nan, nan, nan, nan])

e_mc
array([2.33645016e-15, 1.00191298e+00, nan, nan, nan])

--- elapsed time ---

t_pc
array([0.10312839, 0.28837845, 0.56796813, 0.93851085, 1.39571856])

t_sp
array([0.09956538, 0.28490521, 0.56568614, 0.92945881, 1.38998819])

t_aPC
array([0.00138731, 0.00389138, 0.00794679, 0.01504299, 0.02304922])

t_hd
array([0.00267659, 0.00912098, 0.01920455, 0.03331917, 0.05163501])

t_mc
array([0.00149915, 0.00621344, 0.01429569, 0.0254896 , 0.03943479])


case pws2 (gm = -1, p = q = -1/2)


case pws3 (gm = 1, p = q = 1/2)


case pws4 (gm = -1, p = q = 1/2)



"""


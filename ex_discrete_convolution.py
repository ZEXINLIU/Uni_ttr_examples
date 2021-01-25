import numpy as np

from UncertainSCI.ttr.compute_ttr import predict_correct_discrete, stieltjes_discrete, \
        aPC, hankel_deter, mod_cheb, lanczos_stable

from UncertainSCI.utils.compute_moment import compute_moment_discrete
from UncertainSCI.families import JacobiPolynomials

from UncertainSCI.utils.verify_orthonormal import verify_orthonormal

import time
from tqdm import tqdm

import pdb

"""
We use six methods and use Lanczos as the true solution

1. pc (Predictor-corrector method)
2. sp (Stieltjes procedure)
3. apc (Arbitrary polynomial chaos expansion method)
4. hd (Hankel determinants)
5. mc (Modified Chebyshev algorithm)
6. lz (Stabilized Lanczos algorithm)

to compute the recurrence coefficients for the discrete probability density function.
"""

def preprocess_a(a):
    """
    If a_i = 0 for some i, then the corresponding x_i has no influence
    on the model output and we can remove this variable.
    """
    a = a[np.abs(a) > 0.]
    
    return a

def compute_u(a, N):
    """
    Given the vector a \in R^m (except for 0 vector),
    compute the equally spaced points {u_i}_{i=0}^N-1
    along the one-dimensional interval

    Return
    (N,) numpy.array, u = [u_0, ..., u_N-1]
    """
    # assert N % 2 == 1
    
    a = preprocess_a(a = a)
    u_l = np.dot(a, np.sign(-a))
    u_r = np.dot(a, np.sign(a))
    u = np.linspace(u_l, u_r, N)

    return u

def compute_q(a, N):
    """
    Given: an vector a \in R^m (except for 0 vector),
    compute the discrete approximation to the convolution
    q(u) = (p_0 * p_1 * ...)(u) = \int p_0(t) p_1(u-t) ... dt
    where x_i ~ UNIF[-1,1], i.e. p_i = 1/2 if |x_i|<=1 or 0 o.w.

    Returns
    (N,) numpy.array, q = [q_0, ..., q_N-1]
    """
    u = compute_u(a = a, N = N)
    q = np.zeros(u.shape)
    q[np.abs(u) <= np.abs(a[0])] = 1 / (2 * np.abs(a[0]))
    if len(a) == 1:
        return q

    for i in range(1, len(a)):
        disc_q = np.zeros(u.shape)
        for j in range(N):
            p = np.zeros(u.shape)
            p[np.abs(u[j] - u) <= np.abs(a[i])] = 1 / (2 * np.abs(a[i]))
            disc_q[j] = np.trapz(y = q*p, x = u)
        q = disc_q
    return q

def compute_q_01(a, N):
    """
    Given: an vector a \in R^m (except for 0 vector),
    compute the discrete approximation to the convolution
    q(u) = (p_0 * p_1 * ...)(u) = \int p_0(t) p_1(u-t) ... dt
    where x_i ~ UNIF[0,1], i.e. p_i = 1 if 0<=x_i<=1 or 0 o.w.

    Returns
    (N,) numpy.array, q = [q_0, ..., q_N-1]
    """
    u = compute_u(a = a, N = N)
    q = np.zeros(u.shape)
    q[(0<=u)&(u<=a[0])] = 1 / a[0]
    if len(a) == 1:
        return q

    for i in range(1, len(a)):
        disc_q = np.zeros(u.shape)
        for j in range(N):
            p = np.zeros(u.shape)
            p[(0<=u[j]-u)&(u[j]-u<=a[i])] = 1 / a[i]
            disc_q[j] = np.trapz(y = q*p, x = u)
        q = disc_q
    return q


m = 25
np.random.seed(1)
a = np.random.rand(m,) * 2 - 1.
a = a / np.linalg.norm(a, None) # normalized a

N_quad = 100 # number of discrete univariable u
u = compute_u(a = a, N = N_quad)
du = (u[-1] - u[0]) / (N_quad - 1)

q = compute_q(a = a, N =  N_quad)
w = du*q

N_array = [20, 40, 60, 80, 100]

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

        m = compute_moment_discrete(u, w, N)
        
        # Predict-correct
        start = time.time()
        ab_pc = predict_correct_discrete(u, w, N)
        end = time.time()
        t_pc[ind] += (end - start) / len(iter_n)
        e_pc[ind] = np.linalg.norm(verify_orthonormal(ab_pc, np.arange(N), u, w) \
                - np.eye(N), None)

        # Stieltjes
        start = time.time()
        ab_sp = stieltjes_discrete(u, w, N)
        end = time.time()
        t_sp[ind] += (end - start) / len(iter_n)
        e_sp[ind] = np.linalg.norm(verify_orthonormal(ab_sp, np.arange(N), u, w) \
                - np.eye(N), None)

        # Arbitrary Polynomial Chaos Expansion
        start = time.time()
        ab_apc = aPC(m, N)
        end = time.time()
        t_apc[ind] += (end - start) / len(iter_n)
        e_apc[ind] = np.linalg.norm(verify_orthonormal(ab_apc, np.arange(N), u, w) \
                - np.eye(N), None)

        # Hankel Determinant
        start = time.time()
        ab_hd = hankel_deter(N, m)
        end = time.time()
        t_hd[ind] += (end - start) / len(iter_n)
        e_hd[ind] = np.linalg.norm(verify_orthonormal(ab_hd, np.arange(N), u, w) \
                - np.eye(N), None)

        # Modified Chebyshev
        J = JacobiPolynomials(probability_measure=False)
        peval = lambda x, n: J.eval(x, n)
        mod_m = np.zeros(2*N - 1)
        for i in range(2*N - 1):
            integrand = lambda x: peval(x,i).flatten()
            mod_m[i] = np.sum(integrand(u) * w)
        start = time.time()
        ab_mc = mod_cheb(N, mod_m, J)
        end = time.time()
        t_mc[ind] += (end - start) / len(iter_n)
        e_mc[ind] = np.linalg.norm(verify_orthonormal(ab_mc, np.arange(N), u, w) \
                - np.eye(N), None)
        
        # Stabilized Lanczos
        start = time.time()
        ab_lz = lanczos_stable(u, w, N)
        end = time.time()
        t_lz[ind] += (end - start) / len(iter_n)
        e_lz[ind] = np.linalg.norm(verify_orthonormal(ab_lz, np.arange(N), u, w) \
                - np.eye(N), None)

        
"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 100

--- Frobenius norm error ---

e_pc
array([3.95937126e-15, 1.17465968e-14, 1.03274678e-09, 4.00000000e+00,
       7.48331477e+00])

e_sp
array([2.73718732e-15, 9.26234216e-15, 5.94278038e-10, 4.00000000e+00,
       7.48331477e+00])

e_apc
array([7.71740426e-08, 2.02884310e+05, 1.34663120e+27, 5.84702012e+47,
       4.75461302e+67])

e_hd
array([1.69002753e-07,            nan,            nan,            nan,
                  nan])

e_mc
array([3.02198881e-09,            nan,            nan,            nan,
                  nan])

e_lz
array([4.75363731e-15, 2.95344383e-14, 3.45264757e-09, 1.86341337e+68,
       2.49535465e+68])


--- elapsed time ---

t_pc
array([0.006624  , 0.02360435, 0.04991234, 0.08732417, 0.13431059])

t_sp
array([0.00610083, 0.0226834 , 0.04878942, 0.08529013, 0.13456016])

t_apc
array([0.0012863 , 0.00377234, 0.00772418, 0.01319587, 0.02014426])

t_hd
array([0.00285909, 0.00948635, 0.02003272, 0.03506948, 0.05553345])

t_mc
array([0.001672  , 0.00677977, 0.01486643, 0.02690948, 0.04083029])

t_lz
array([0.00102357, 0.00202671, 0.00326769, 0.00459507, 0.00605136])


N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 200

e_pc
array([3.11798041e-15, 7.39256700e-15, 1.29863095e-14, 9.88664797e-14,
       3.61285380e-10])

e_sp
array([3.72126779e-15, 7.53592850e-15, 1.26781374e-14, 2.20923974e-13,
       9.87486694e-10])

e_apc
array([7.78482843e-09,            nan,            nan,            nan,
                  nan])

e_hd
array([1.32862308e-07,            nan,            nan,            nan,
                  nan])

e_mc
array([1.35810897e-08,            nan,            nan,            nan,
                  nan])

e_lz
array([3.41140713e-15, 1.14879579e-14, 2.11319394e-14, 1.59865492e-13,
       6.02815932e-10])

t_pc
array([0.00723969, 0.02657084, 0.05732713, 0.10112422, 0.15605735])

t_sp
array([0.00674671, 0.02566374, 0.05539571, 0.0994516 , 0.15175851])

t_apc
array([0.00124123, 0.00373606, 0.00757654, 0.01280226, 0.01998155])

t_hd
array([0.00280116, 0.00920861, 0.01995641, 0.03663265, 0.05622729])

t_mc
array([0.0016241 , 0.00665073, 0.01525463, 0.02681506, 0.04088025])

t_lz
array([0.00116924, 0.00255765, 0.00427727, 0.00629655, 0.00895607])


N_array = [20, 40, 60, 80, 100] with tol = 1e-12, N_quad = 300

e_pc
array([4.53766132e-15, 9.56792320e-15, 1.49645839e-14, 2.46683586e-14,
       1.40823522e-13])
e_sp
array([3.38625397e-15, 8.53221733e-15, 2.61509605e-14, 5.20440256e-14,
       9.37261060e-14])

e_apc
array([3.79788608e-08, 7.66674623e+05, 3.90076936e+25, 2.70711113e+55,
       9.69092414e+72])

e_hd
array([1.23521956e-07,            nan,            nan,            nan,
                  nan])

e_mc
array([3.59601296e-09,            nan,            nan,            nan,
                  nan])

e_lz
array([3.86864436e-15, 1.09695006e-14, 1.73436385e-14, 3.37621255e-14,
       9.29458751e-14])

t_pc
array([0.0078077 , 0.02890715, 0.06318885, 0.11134079, 0.17054021])

t_sp
array([0.00751934, 0.02812571, 0.06108396, 0.10756618, 0.17055254])

t_apc
array([0.00134934, 0.00386411, 0.0075922 , 0.01308334, 0.02020775])

t_hd
array([0.00286912, 0.00990186, 0.0207794 , 0.03607668, 0.05730121])

t_mc
array([0.00165269, 0.00695888, 0.01558213, 0.02764288, 0.04210448])

t_lz
array([0.00145459, 0.00330329, 0.00638815, 0.01078056, 0.01479507])

"""

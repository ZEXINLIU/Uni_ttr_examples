import numpy as np
from UncertainSCI.ttr import lanczos_stable
from UncertainSCI.ttr import predict_correct_unbounded
from UncertainSCI.opoly1d import gauss_quadrature_driver
from UncertainSCI.utils.verify_orthonormal import verify_orthonormal

a = 0.
b = np.inf


def weight(x):
    return np.exp(-x**2)


singularity_list = []

N_array = [20, 40, 60, 80, 100]
M = 320

eps = 1e-12

e_pcl = np.zeros(len(N_array),)

adaptive = True

for ind, N in enumerate(N_array):

    if not adaptive:
        ab_pc = predict_correct_unbounded(a, b, weight, N+1, singularity_list)
        xc, wc = gauss_quadrature_driver(ab_pc, N)
        xd = -np.arange(M) / M
        wd = (1/M) * np.ones(len(xd))
        xg = np.hstack([xd[::-1], xc])
        wg = np.hstack([wd, wc])
        ab = lanczos_stable(xg, wg, N)
        e_pcl[ind] = np.linalg.norm(verify_orthonormal(ab, np.arange(N),
                                    xg, wg) - np.eye(N), None)
    else:
        i = 0
        K0 = N
        ab_pc = predict_correct_unbounded(a, b, weight, K0+1, singularity_list)
        xc, wc = gauss_quadrature_driver(ab_pc, K0)
        xd = -np.arange(M) / M
        wd = (1/M) * np.ones(len(xd))
        xg = np.hstack([xd[::-1], xc])
        wg = np.hstack([wd, wc])
        ab = lanczos_stable(xg, wg, N)

        i = 1
        K = K0 + 1
        ab_pc = predict_correct_unbounded(a, b, weight, K+1, singularity_list)
        xc, wc = gauss_quadrature_driver(ab_pc, K)
        xd = -np.arange(M) / M
        wd = (1/M) * np.ones(len(xd))
        xg = np.hstack([xd[::-1], xc])
        wg = np.hstack([wd, wc])
        ab_new = lanczos_stable(xg, wg, N)

        while any(np.abs(ab[:, 1] - ab_new[:, 1]) >
                  eps * np.abs(ab_new[:, 1])) and K < 150:
            ab = ab_new
            i += 1
            K = K + int(2**(np.floor(i/5)) * N)
            ab_pc = predict_correct_unbounded(a, b, weight, K+1,
                                              singularity_list)
            xc, wc = gauss_quadrature_driver(ab_pc, K)
            xd = -np.arange(M) / M
            wd = (1/M) * np.ones(len(xd))
            xg = np.hstack([xd[::-1], xc])
            wg = np.hstack([wd, wc])
            ab_new = lanczos_stable(xg, wg, N)

        e_pcl[ind] = np.linalg.norm(verify_orthonormal(ab_new, np.arange(N),
                                    xg, wg) - np.eye(N), None)

"""
N_array = [20, 40, 60, 80, 100] with tol = 1e-12

M = 20

array([1.08770625e-14, 6.48170205e-14, 1.45988558e-10, 2.41144892e-03,
       1.66400897e+07])

array([7.46641452e-15, 1.63349266e-14, 6.61412176e-13, 5.63466698e-12,
       3.27237243e-09])


M = 40

array([6.50495608e-15, 2.49905213e-14, 9.34168638e-11, 8.53863949e-03,
       1.95102968e+09])

array([1.04852833e-14, 3.28118597e-14, 9.51591941e-14, 1.84055514e-13,
       3.05482540e-11])


M = 80

array([8.80415627e-15, 1.38833090e-14, 1.67517945e-11, 4.48309576e-03,
       5.09980339e+08])

array([5.11454090e-15, 4.74387244e-14, 3.90481756e-14, 8.96559954e-14,
       4.95432711e-11])


M = 160

array([7.72910025e-15, 1.42816844e-14, 2.90067523e-11, 1.87516001e-03,
       2.34023152e+09])

array([7.12640234e-15, 3.99054264e-14, 7.03130755e-14, 1.24323161e-13,
       2.25212929e-11])

M = 320

array([7.24409273e-15, 1.97732242e-14, 3.79735052e-11, 6.81895017e-03,
       9.53351221e+08])

array([8.38882081e-15, 1.67724423e-14, 3.85803156e-14, 6.64530027e-14,
       7.13773984e-11])
"""

import numpy as np
from UncertainSCI.ttr.compute_ttr import dPI4, dPI6, hankel_deter, mod_cheb, aPC
from UncertainSCI.utils.compute_moment import compute_freud_moment
from UncertainSCI.utils.quad import gq_modification_unbounded_composite
from UncertainSCI.families import HermitePolynomials
from matplotlib import pyplot as plt

N = 35

a = -np.inf
b = np.inf
singularity_list = []


fig,axs = plt.subplots(2,2)
m = compute_freud_moment(rho = 0, m = 4, n = N)
ab_hd = hankel_deter(N, m)
weight = lambda x: np.exp(-x**4)
H = HermitePolynomials(probability_measure=False)
peval = lambda x, n: H.eval(x, n)
mod_m = np.zeros(2*N - 1)
for i in range(2*N - 1):
    integrand = lambda x: weight(x) * peval(x,i).flatten()
    mod_m[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
ab_mc = mod_cheb(N, mod_m, H)
ab_apc = aPC(m, N)

axs[0,0].plot(np.arange(1,N), dPI4(N)[1:,1], 'x', label = 'DP')
axs[0,0].plot(np.arange(1,N), ab_hd[1:,1], 'o', markerfacecolor="None", label = 'HD')
axs[0,0].plot(np.arange(1,N), ab_apc[1:,1], '1', label = 'aPC')
axs[0,0].plot(np.arange(1,N), ab_mc[1:,1], '+', label = 'MC')
axs[0,0].plot(np.arange(1,N), (np.arange(1,N)/12)**(1/4), '--', label = 'Conjecture')
axs[0,0].set_xlabel(r'$N$')
axs[0,0].set_ylabel(r'$b_N$')
axs[0,0].set_ylim(0,3)
axs[0,0].set_yticks(np.arange(0,3,1))
axs[0,0].set_title(r'$\alpha = 4$')
axs[0,0].legend(prop={'size':7})


m = compute_freud_moment(rho = 0, m = 6, n = N)
ab_hd = hankel_deter(N, m)
weight = lambda x: np.exp(-x**6)
H = HermitePolynomials(probability_measure=False)
peval = lambda x, n: H.eval(x, n)
mod_m = np.zeros(2*N - 1)
for i in range(2*N - 1):
    integrand = lambda x: weight(x) * peval(x,i).flatten()
    mod_m[i] = gq_modification_unbounded_composite(integrand, a, b, 10, singularity_list)
ab_mc = mod_cheb(N, mod_m, H)
ab_apc = aPC(m, N)

axs[0,1].plot(np.arange(1,N), dPI6(N)[1:,1], 'x', label = 'DP')
axs[0,1].plot(np.arange(1,N), ab_hd[1:,1], 'o', markerfacecolor="None", label = 'HD')
axs[0,1].plot(np.arange(1,N), ab_apc[1:,1], '1', label = 'aPC')
axs[0,1].plot(np.arange(1,N), ab_mc[1:,1], '+', label = 'MC')
axs[0,1].plot(np.arange(1,N), (np.arange(1,N)/60)**(1/6), '--', label = 'Conjecture')
axs[0,1].set_xlabel(r'$N$')
axs[0,1].set_ylabel(r'$b_N$')
axs[0,1].set_ylim(0, 3)
axs[0,1].set_yticks(np.arange(0,3,1))
axs[0,1].set_title(r'$\alpha = 6$')


N_array = [20, 40, 60, 80, 100]
e_sp = np.array([8.91113940e-15, 1.42689811e-14, 3.02075444e-14, 5.76568141e-14, 8.63063396e-14])
e_pc = np.array([3.10475162e-15, 5.38559426e-15, 9.24185904e-15, 1.11959041e-14, 1.33658569e-14])
axs[1,0].semilogy(N_array, e_sp, '-^', label = 'SP')
axs[1,0].semilogy(N_array, e_pc, '-o', label = 'PC')
axs[1,0].set_xlabel(r'$N$')
axs[1,0].set_ylabel(r'$e_N$')
axs[1,0].legend(prop={'size':7})

e_sp = np.array([1.67486852e-14, 3.94620768e-14, 5.85110601e-14, 1.08525933e-13, 1.37666177e-13])
e_pc = np.array([2.59281844e-14, 2.64537346e-14, 2.68816085e-14, 2.75287150e-14, 2.81177506e-14])
axs[1,1].semilogy(N_array, e_sp, '-^', label = 'SP')
axs[1,1].semilogy(N_array, e_pc, '-o', label = 'PC')
axs[1,1].set_xlabel(r'$N$')
axs[1,1].set_ylabel(r'$e_N$')

plt.tight_layout()
plt.show()

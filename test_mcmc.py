"""
Tiny standalone script: Testing and How to use
Constructor arg (sigma) of CauchyRandomWalk_1D() controls accept prob (acf decay)
"""

import numpy as np
import matplotlib.pyplot as plt

import mcmc

plt.rcParams['patch.force_edgecolor'] = True
plt.close('all')

A = mcmc.CauchyRandomWalk_1D(sigma=3.)
A._state = 200. # initialize far from equilibrium (check convergence)

# Markow random walk (marginal pdf: Cauchy)
N = 2 * 10**5
x = A.step_n(N)
p_accept = A.get_p_accept()

# Auto-correlation
K = 200 # number of one-sided lags (excluding zero)
acf = np.zeros(K + 1 + K)
for k in range(K+1):
    acf[K+k] = x[k:N].dot(x[0:N-k]) / (N-k)
acf[0:K] = acf[K+1:2*K+1][::-1] # reversal
acf[:] *= 1./acf[K] # normalize to unity at lag zero

# Mapped/transformed random walk (marginal pdf: uniform)
u = (np.arctan(x) + np.pi/2)/np.pi # map to the unit interval (0.,1.)

# Figures
fig,ax = plt.subplots()
ax.set_title('MC realization (p_accept = {:.1f}%)'.format(100.*p_accept))
ax.plot(x)

fig,ax = plt.subplots()
ax.set_title('Cauchy density')
ax.hist(x, bins=501, density=True)
halfwidth = 15.
grid = np.linspace(-halfwidth, halfwidth, 201)
ax.plot(grid, (1./np.pi)/(1.+grid**2), 'r')
ax.set_xlim([-halfwidth, halfwidth])

fig,ax = plt.subplots()
ax.set_title('Autocorrelation (normalized)')
step = 5
ax.stem(np.arange(-K,K+1,step), acf[::step], use_line_collection=True)

fig,ax = plt.subplots()
ax.set_title('Mapped/transformed realization')
ax.plot(u)

fig,ax = plt.subplots()
ax.set_title('Standard uniforms')
ax.hist(u, bins=101, density=True)

plt.show(block=False)

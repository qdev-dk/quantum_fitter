import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

def fit_strong(x, p):
    return 1 / ((1 - p) + p / x)

def fit_weak(x, p):
    return 1-p + p*x

n1x = np.array([1,2,4])
n1read = np.array([2.2693,1.592666,1.45723])
n1time = np.array([5.65493, 4.27359, 4.22066])

n2x = np.array([2,4])
n2read = np.array([1.23772,1.04316])
n2time = np.array([2.59609, 3.72961])

n4x = n1x
n4read = np.array([1.2035, 0.802488,0.683511])  # 1.02295 for 16 threads
n4time = np.array([2.40671, 1.70918,1.64663])  # 2.16116


n8x=n2x
n8read = np.array([0.940072,0.837491])
n8time= np.array([1.46459,1.51594])

plt.figure()
plt.title("Benchmark within one node")
plt.xlabel('Number of MPI Processes')
plt.xticks(n1x)
plt.ylabel('Speedup t1/tN')
plt.grid(linestyle='--')
plt.plot(n1x, n1read[0]/n1read, c='b', label='Reading')
plt.plot(n1x, n1time[0]/n1time, c='r', ls='--', label='Total')
plt.legend()

plt.figure()
plt.title("Benchmark within different node")
plt.xlabel('Number of MPI Processes per node')
plt.xticks(n1x)
plt.ylabel('Reading time / sec')
plt.grid(linestyle='--')
plt.plot(n1x, n1read, c='b', label='n1')
plt.plot(n2x, n2read, c='r', label='n2')
plt.plot(n4x, n4read, c='y', label='n4')
plt.plot(n8x, n8read, c='c', label='n8')
plt.legend()

plt.figure()
plt.title("Benchmark by MPI processes")
plt.xlabel('Number of MPI Processes')
nx = np.concatenate((n1x[:2], n4x[:]*4, n8x[1:]*8))
ny = n1read[0]/np.concatenate((n1read[:2], n4read[:], n8read[1:]))
ntt = n1time[0]/np.concatenate((n1time[:2], n4time[:], n8time[1:]))
poptr, pcovr = curve_fit(fit_strong, nx, ny)
poptt, pcovt = curve_fit(fit_strong, nx, ntt)

plt.xticks(nx)
plt.xscale('log', basex=2)
plt.ylabel('Speed up T0/Tn')
plt.grid(linestyle='--')
plt.plot(nx, ny, c='b', label='Speed up of Reading')
plt.plot(nx, ntt, c='c',ls='--', label='Speed up of Execution')
plt.plot(nx, fit_strong(nx, poptr[0]), c='r', ls='-.', label=('Fit with Amdahl’s law (reading), p = '+str(poptr[0])[:4]))
plt.plot(nx, fit_strong(nx, poptt[0]), c='m', ls=':', label=('Fit with Amdahl’s law (exe), p = '+str(poptt[0])[:4]))
plt.legend()

plt.figure()
plt.title("Benchmark with Weak scaling")
nx = np.array([1, 4, 16])
nyr = np.array([2.20604, 0.900655, 1.03944])
nyr = nyr[0]/nyr
nye = np.array([2.99662, 2.2372, 8.03817])
nye = nye[0] / nye
poptr, pcovr = curve_fit(fit_weak, nx, nyr[0]/nyr)
poptt, pcovt = curve_fit(fit_weak, nx, nye[0]/nye)
plt.xticks(nx)
plt.xscale('log', basex=2)
plt.ylabel('Speed up T0/Tn')
plt.grid(linestyle='--')
plt.plot(nx, nyr, c='b', label='Speed up of Reading')
plt.plot(nx, nye, c='c', ls='--', label='Speed up of Execution')
plt.plot(nx, fit_weak(nx, poptr[0]), c='r', ls='-.', label=('Fit with Gustafson’s law (reading), p = '+str(poptr[0])[:4]))
plt.plot(nx, fit_weak(nx, poptt[0]), c='m', ls=':', label=('Fit with Gustafson’s law (exe), p = '+str(poptt[0])[:4]))

plt.legend()
plt.show()

import qutip as qt
from qutip import dag
import numpy as np
import matplotlib.pyplot as plt

N_c = 30
N_t = 30
alpha = 3.5
beta = 3.5
K = 2 * np.pi * 0.005
T = 100
itr = 2000
phi_dot = 0.001

ac = qt.tensor(qt.destroy(N_c), qt.qeye(N_t))
at = qt.tensor(qt.qeye(N_c), qt.destroy(N_t))

beta_ket = qt.coherent(N=N_c, alpha=alpha)
min_beta_ket = qt.coherent(N=N_c, alpha=-alpha)
alpha_ket = qt.coherent(N=N_t, alpha=beta)
min_alpha_ket = qt.coherent(N=N_t, alpha=-beta)

ac_plus = (beta + ac) / (2 * beta)
ac_min = (beta - ac) / (2 * beta)
acd_plus = (beta + dag(ac)) / (2 * beta)
acd_min = (beta - dag(ac)) / (2 * beta)

# alpha is target, beta is control
Hc_0 = -K * ((dag(ac) ** 2) - (beta ** 2)) * ((ac ** 2) - (beta ** 2))
H_phidot = -(phi_dot / (4 * beta)) * dag(at) * at * (2 * beta - dag(ac) - ac)
H_int_NTD = -K * ((dag(at) ** 2) * (at ** 2) -
                  (alpha ** 2) * (dag(at) ** 2) * ac_plus - (alpha ** 2) * acd_plus * (at ** 2) +
                  (alpha ** 4) * (acd_plus * ac_plus + acd_min * ac_min))
H_int_TD_min = K * (alpha ** 2) * acd_min * (at ** 2 - (alpha ** 2) * ac_plus)
H_int_TD_plus = K * (alpha ** 2) * (dag(at) ** 2 - (alpha ** 2) * acd_plus) * ac_min


def coeff_min(t, args):
    return np.exp(-2 * 1j * np.pi * t / T)


def coeff_plus(t, args):
    return np.exp(2 * 1j * np.pi * t / T)


t_series = np.linspace(start=0, stop=T, num=itr)

H = [Hc_0, H_int_NTD, [H_int_TD_min, coeff_min], [H_int_TD_plus, coeff_plus], H_phidot]
psi0 = qt.tensor(min_beta_ket, alpha_ket)
res = qt.sesolve(H=H, psi0=psi0, tlist=t_series)
count = 0
for state in res.states:
    if count % 200 == 0:
        qt.plot_wigner(qt.ptrace(state, 1), alpha_max=10)
        plt.show()
    count = count + 1

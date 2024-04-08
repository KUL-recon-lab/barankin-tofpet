# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar

from collections.abc import Callable

from pdfs import truncated_double_gauss_pdf
from utils import U_N_ij


# %%
# input parameters
num_possible_deltas: int = 200  # number of possible deltas
delta_min: float = 0.001  # max delta value
delta_max: float = 20.0  # max delta value
delta_mode: str = "log"  # delta mode: 'log' or 'lin'
N: int = 10  # number of photons
Jmax: int = 32
upper_int_limit = None

rcond: float = 1e-8  # fraction of largest singular value for pinv (rcond)

# choice of the user defined pdf
pdf: Callable[[float], float] = truncated_double_gauss_pdf

# delta array

if delta_mode == "lin":
    all_possible_deltas = np.linspace(delta_min, delta_max, num_possible_deltas)
else:
    all_possible_deltas = np.logspace(
        np.log10(delta_min), np.log10(delta_max), num_possible_deltas
    )

# %%
# estimate upper integration limit (point beyond which pdf is essentially zero + delta_max)

if upper_int_limit is None:
    x_zero = root_scalar(lambda x: pdf(x) - 1e-8, x0=10, bracket=[0, 1e6]).root
    upper_int_limit = x_zero + delta_max

# plot the pdf and the upper integration limit
tt = np.linspace(-1, upper_int_limit, 1000)
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
ax.plot(tt, [pdf(x) for x in tt])
ax.set_title(f"pdf")
ax.grid(ls=":")
fig.show()


# %%
# calculate max Barankin bound when choosing 1 delta
test_bbs = np.zeros(num_possible_deltas)
U_Ns = []

for j, delta in enumerate(all_possible_deltas):
    test_deltas = np.array([delta])
    U_N = np.array([[U_N_ij(pdf, delta, delta, N, upper_int_limit)]])
    U_Ns.append(U_N)

    test_bbs[j] = test_deltas.T @ (np.linalg.pinv(U_N, rcond=rcond) @ test_deltas)

# %%
# picks deltas step by step starting from J=1 case
i_deltas = np.array([np.argmax(test_bbs)])
U_cur = U_Ns[i_deltas[0]]

for J in range(Jmax - 1):
    nd = i_deltas.shape[0]
    U_Ns = []
    test_bbs = np.zeros(all_possible_deltas.shape[0])

    for j in range(all_possible_deltas.shape[0]):
        U_N = np.zeros((nd + 1, nd + 1))
        U_N[:nd, :nd] = U_cur

        for k, i_del in enumerate(i_deltas):
            U_N[k, -1] = U_N_ij(
                pdf,
                all_possible_deltas[i_del],
                all_possible_deltas[j],
                N,
                upper_int_limit,
            )
            U_N[-1, k] = U_N[k, -1]

        U_N[-1, -1] = U_N_ij(
            pdf, all_possible_deltas[j], all_possible_deltas[j], N, upper_int_limit
        )

        U_Ns.append(U_N)
        test_deltas = all_possible_deltas[np.concatenate((i_deltas, [j]))]
        test_bbs[j] = test_deltas.T @ (np.linalg.pinv(U_N, rcond=rcond) @ test_deltas)

    i_new = np.argmax(test_bbs)
    if i_new == (num_possible_deltas - 1):
        raise ValueError("Hit upper bound of deltas. Stopping")
    elif i_new == 0:
        raise ValueError("Hit lower bound of deltas. Stopping")

    i_deltas = np.concatenate((i_deltas, [i_new]))
    U_cur = U_Ns[i_new]
    print(f"{(J + 2):04}  {test_bbs.max():.4E}  {i_deltas}")

    # fig3, ax3 = plt.subplots(figsize=(6, 6), tight_layout=True)
    # ax3.plot(test_bbs)
    # fig3.show()
    # plt.close()

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar
from scipy.integrate import quad

from collections.abc import Callable

from pdfs import truncated_double_gauss_pdf, exp_pdf
from utils import U_N_ij


# %%
# input parameters
num_possible_deltas: int = 200  # number of possible deltas
delta_min: float | None = None  # max delta value
delta_max: float | None = None  # max delta value
delta_mode: str = "log"  # delta mode: 'log' or 'lin'
N: int = 1  # number of photons
Jmax: int = 50
x_zero: float | None = None  # point beyond which pdf is essentially zero

rcond: float = 1e-8  # fraction of largest singular value for pinv (rcond)

# choice of the user defined pdf
pdf: Callable[[float], float] = truncated_double_gauss_pdf
# pdf: Callable[[float], float] = exp_pdf

# show interactive plots
interactive: bool = False

# %%
# estimate delta_min and delta_max
if x_zero is None:
    x_zero = root_scalar(lambda x: pdf(x) - 1e-8, x0=10, bracket=[0, 1e6]).root

# normalize the pdf
norm = quad(pdf, 0, x_zero)[0]
normalized_pdf = lambda x: pdf(x) / norm

stddev_pdf = np.sqrt(
    quad(lambda x: (x**2) * normalized_pdf(x), 0, x_zero)[0]
    - quad(lambda x: x * normalized_pdf(x), 0, x_zero)[0] ** 2
)

if delta_max is None:
    delta_max = 20.0 * stddev_pdf / N
if delta_min is None:
    delta_min = 0.003 * stddev_pdf / N

print(f"delta_min: {delta_min:.2E}")
print(f"delta_max: {delta_max:.2E}")

# %%
# delta array
if delta_mode == "lin":
    all_possible_deltas: list[float] = np.linspace(
        delta_min, delta_max, num_possible_deltas
    )
else:
    all_possible_deltas: list[float] = np.logspace(
        np.log10(delta_min), np.log10(delta_max), num_possible_deltas
    )

available_delta_inds = np.arange(num_possible_deltas).tolist()
chosen_delta_inds = []

# %%
upper_int_limit = x_zero + delta_max

# plot the pdf and the upper integration limit
tt = np.linspace(-x_zero / 10, x_zero, 1000)
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
ax.plot(tt, [normalized_pdf(x) for x in tt])
ax.set_title(f"normalized pdf")
ax.grid(ls=":")
fig.show()


# %%
# calculate max Barankin bound when choosing 1 delta
test_bbs = np.zeros(num_possible_deltas)
U_Ns = []

U_N_ij_lut = dict()

for j, delta in enumerate(all_possible_deltas):
    test_deltas = np.array([delta])
    U_N = np.array([[U_N_ij(normalized_pdf, delta, delta, N, upper_int_limit)]])
    U_N_ij_lut[(j, j)] = U_N[0, 0]
    U_Ns.append(U_N)

    test_bbs[j] = test_deltas.T @ (np.linalg.pinv(U_N, rcond=rcond) @ test_deltas)

# %%
# picks deltas step by step starting from J=1 case
i_delta_max = np.argmax(test_bbs)
if i_delta_max == (num_possible_deltas - 1):
    raise ValueError("Hit upper bound of deltas. Increase upper bound and rerun.")
elif i_delta_max == 0:
    raise ValueError("Hit lower bound of deltas. Decrease lower bound and rerun.")

U_cur = U_Ns[i_delta_max]
chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))

bbs = []

for J in range(Jmax - 1):
    nd = len(chosen_delta_inds)
    U_Ns = []
    test_bbs = np.zeros(len(available_delta_inds))

    for i_j, j in enumerate(available_delta_inds):
        delta = all_possible_deltas[j]
        U_N = np.zeros((nd + 1, nd + 1))
        U_N[:nd, :nd] = U_cur

        for k, i_del in enumerate(chosen_delta_inds):
            i1 = min(j, i_del)
            i2 = max(j, i_del)

            if (i1, i2) in U_N_ij_lut:
                U_N[k, -1] = U_N_ij_lut[(i1, i2)]
            else:
                U_N[k, -1] = U_N_ij(
                    normalized_pdf,
                    all_possible_deltas[i_del],
                    delta,
                    N,
                    upper_int_limit,
                )
                U_N_ij_lut[(i1, i2)] = U_N[k, -1]

            U_N[-1, k] = U_N[k, -1]

        U_N[-1, -1] = U_N_ij_lut[(j, j)]

        U_Ns.append(U_N)
        test_deltas = all_possible_deltas[np.concatenate((chosen_delta_inds, [j]))]
        test_bbs[i_j] = test_deltas.T @ (np.linalg.pinv(U_N, rcond=rcond) @ test_deltas)

    i_delta_max = np.argmax(test_bbs)

    if available_delta_inds[i_delta_max] == (num_possible_deltas - 1):
        raise ValueError("Hit upper bound of deltas. Increase upper bound and rerun.")
    elif available_delta_inds[i_delta_max] == 0:
        raise ValueError("Hit lower bound of deltas. Decrease lower bound and rerun.")

    if interactive:
        figi, axi = plt.subplots(figsize=(8, 4), tight_layout=True)
        axi.plot(available_delta_inds, test_bbs)
        axi.axvline(available_delta_inds[i_delta_max], color="r", ls="--")
        figi.show()
        tmp = input(">")
        plt.close()

    chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))
    U_cur = U_Ns[i_delta_max]
    bbs.append(test_bbs[i_delta_max])
    print(f"{(J + 2):04}  {test_bbs[i_delta_max]:.4E}", end="\r")

print()

fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax3[0].plot(bbs, ".-")
ax3[1].semilogy(all_possible_deltas[chosen_delta_inds], ".-")
ax3[2].semilogy(sorted(all_possible_deltas[chosen_delta_inds]), ".-")
for axx in ax3:
    axx.grid(ls=":")
    axx.set_xlabel("J")
for axx in ax3[1:]:
    axx.axhline(delta_min, color="k", ls="--")
    axx.axhline(delta_max, color="k", ls="--")
ax3[0].set_title("barankin bound")
ax3[1].set_title("deltas")
ax3[2].set_title("sorted deltas")
fig3.show()

# TODO: smarter choice of delta_min and delta_max, upper_int_limit
from __future__ import annotations

import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.integrate import quad
from scipy.special import comb
from scipy.optimize import root_scalar

from collections.abc import Callable


def exp_pdf(x: float, mu: float = 1.0) -> float:
    if x >= 0:
        p = mu * math.exp(-mu * x)
    else:
        p = 0
    return p


def truncated_gauss_pdf(x: float, mu: float = 2.0, sig: float = 3.0) -> float:
    if x >= 0:
        p = math.exp(-0.5 * ((x - mu) / sig) ** 2)
    else:
        p = 0
    return p


def truncated_double_gauss_pdf(
    x: float,
    mu1: float = 7.0,
    sig1: float = 3.0,
    mu2: float = 70.0,
    sig2: float = 60.0,
) -> float:
    if x >= 0:
        p = math.exp(-0.5 * ((x - mu1) / sig1) ** 2) + 0.3 * math.exp(
            -0.5 * ((x - mu2) / sig2) ** 2
        )
    else:
        p = 0
    return p


def biexp_pdf(x: float, mu1: float = 3.0, mu2: float = 0.5, a: float = 0.5) -> float:
    if x >= 0:
        p = a * mu1 * math.exp(-mu1 * x) * (1 - a) * mu2 * math.exp(-mu2 * x)
    else:
        p = 0
    return p


# %%
# input parameters
num_possible_deltas: int = 256  # number of possible deltas
J: int = 128  # number of deltas used for one simulation
delta_min: float = 0.001  # max delta value
delta_max: float | None = None  # max delta value
delta_mode: str = "log"  # delta mode: 'log' or 'lin'

num_sim: int = 500  # number of simulations

num_photons: int = 1  # number of photons

upper_int_limit: float | None = (
    None  # upper integration limit, None means auto determined
)
rcond: float = 1e-8  # fraction of largest singular value for pinv (rcond)

# choice of the user defined pdf
# pdf: Callable[[float], float] = exp_pdf
# pdf: Callable[[float], float] = biexp_pdf
pdf: Callable[[float], float] = truncated_double_gauss_pdf

# %%
x_zero = root_scalar(lambda x: pdf(x) - 1e-8, x0=10, bracket=[0, 1e6]).root

# %%
max_num_sim = comb(num_possible_deltas, J, exact=True)

if num_sim > max_num_sim:
    raise ValueError(
        f"Number of simulations ({num_sim}) is larger than the number of possible combinations ({max_num_sim})"
    )

# %%
# pre-calculate all possible matrix elements U_ij using all num_possible_deltas over 2
# combinations of deltas for the integral eta(t, delta_1) eta(t, delta_2) pdf(t)

eta: Callable[[float, float], float] = lambda t, delta: pdf(t - delta) / pdf(t) - 1
integrand: Callable[[float, float, float], float] = (
    lambda t, a, b: eta(t, a) * eta(t, b) * pdf(t)
)

# check the largest and smallest possible U_ij
integ_min = lambda x: integrand(x, delta_min, delta_min)
Umin = (
    quad(integ_min, 0, delta_min)[0] + quad(integ_min, delta_min, x_zero + delta_min)[0]
)
U_N_min = (Umin + 1) ** num_photons - 1

if delta_max is None:
    print("finding delta_max")
    delta_max = 1.01 * delta_min
    it = 0
    for it in range(100):
        integ_max = lambda x: integrand(x, delta_max, delta_max)
        Umax = (
            quad(integ_max, 0, delta_max)[0]
            + quad(integ_max, delta_max, x_zero + delta_max)[0]
        )
        U_N_max = (Umax + 1) ** num_photons - 1
        log_dyn_range = np.log10(U_N_max / U_N_min)
        print(f"{it:03} {log_dyn_range:.2f}", end="\r")

        if log_dyn_range < 6.0:
            delta_max *= 2
        elif log_dyn_range > 15.0:
            delta_max /= 2
        else:
            break
    print()
else:
    integ_max = lambda x: integrand(x, delta_max, delta_max)
    Umax = (
        quad(integ_max, 0, delta_max)[0]
        + quad(integ_max, delta_max, x_zero + delta_max)[0]
    )
    U_N_max = (Umax + 1) ** num_photons - 1
    log_dyn_range = np.log10(U_N_max / U_N_min)


if upper_int_limit is None:
    upper_int_limit = x_zero + delta_max

print(f"upper integration limit: {upper_int_limit:.2E}")
print(f"delta min / pdf(delta min): {delta_min:.2E} / {pdf(delta_min):.2E}")
print(f"delta max / pdf(delta max): {delta_max:.2E} / {pdf(delta_max):.2E}")
print(f"log dynamic range of all possible U_N_ij: {log_dyn_range:.2f}")

if log_dyn_range > 15.0:
    raise ValueError("Dynamic range of U_N_ij is too large. Decrease delta_max.")
if log_dyn_range < 6.0:
    raise ValueError("Dynamic range of U_N_ij is too low. Increase delta_max.")

if delta_mode == "log":
    all_possible_deltas = np.logspace(
        np.log10(delta_min), np.log10(delta_max), num_possible_deltas
    )
elif delta_mode == "lin":
    all_possible_deltas = np.linspace(delta_min, delta_max, num_possible_deltas)
else:
    raise ValueError("delta_mode must be 'log' or 'lin'")


print(
    f"pre-calculate all possible ({num_possible_deltas*(num_possible_deltas-1)//2}) matrix elements U_ij"
)

all_U_ij = np.zeros((num_possible_deltas, num_possible_deltas))

for i in range(num_possible_deltas):
    for j in range(i + 1):
        integ = lambda x: integrand(x, all_possible_deltas[i], all_possible_deltas[j])
        l1 = all_possible_deltas[j]
        l2 = all_possible_deltas[i]

        if i != j:
            IL = quad(integ, 0, l1)
            IM = quad(integ, l1, l2)
            IR = quad(integ, l2, upper_int_limit)
            val = IL[0] + IM[0] + IR[0]
        else:
            IL = quad(integ, 0, l1)
            IR = quad(integ, l1, upper_int_limit)
            val = IL[0] + IR[0]

        all_U_ij[i, j] = val
        all_U_ij[j, i] = val

# check dynamic range of all possible U_ijs transformed to U_N
all_U_N_ij = (all_U_ij + 1) ** num_photons - 1
figU, axU = plt.subplots(tight_layout=True)
imU = axU.matshow(
    all_U_N_ij,
    norm=LogNorm(vmin=all_U_N_ij.min(), vmax=all_U_N_ij.max()),
)
axU.set_title(f"all possible U_N_ij (delta_min={delta_min}, delta_max={delta_max})")
figU.colorbar(imU)
figU.show()

# %%
np.random.seed(1)

bb = np.zeros(num_sim)
simulated_deltas = np.zeros((num_sim, J))
all_inds = np.arange(num_possible_deltas)

for i_sim in range(num_sim):
    inds = np.random.choice(all_inds, size=(J,))
    inds.sort()

    deltas = all_possible_deltas[inds]
    simulated_deltas[i_sim, :] = deltas

    U = np.zeros((J, J))

    for i, ii in enumerate(inds):
        for j, jj in enumerate(inds):
            U[i, j] = all_U_ij[ii, jj]

    U_N = (U + 1) ** num_photons - 1

    pinv_U_N = np.linalg.pinv(U_N, rcond=rcond)

    bb[i_sim] = float(np.sum(deltas * (pinv_U_N @ deltas)))
    print(f"{(i_sim+1):04}/{num_sim:04}, {bb[i_sim]:.2E}", end="\r")

print()

# %%
# sort all bounds and used deltas

i_sort = np.argsort(bb)
bb_sorted = bb[i_sort]
simulated_deltas_sorted = simulated_deltas[i_sort, :]

print(f"highest Barankin bound (BB)          ..: {bb_sorted[-1]:.2E}")
print(f"lowest  Barankin bound (BB)          ..: {bb_sorted[0]:.2E}")
print(f"5% percentile of all Barankin bounds ..: {bb_sorted[int(0.05*num_sim)]:.2E}")

if np.isclose(simulated_deltas_sorted[-1].max(), delta_max):
    warnings.warn(
        "WARNING: Simulation that that yielded highest bound hit max delta value. Increase delta_max"
    )

# see how far the lowest 5% of the simulation are from the max bound
rdist = 1 - bb_sorted[int(0.05 * num_sim)] / bb_sorted[-1]
print(f"95% of the simulated BBs are within {int(100*rdist)}% of the max BB")

if rdist > 0.2:
    warnings.warn(
        "WARNING: lower 5% of the simulated BBs are more than 20% away from the max BB. Increase J"
    )

# %%
fig3, ax3 = plt.subplots(
    2,
    2,
    figsize=(8, 8),
    tight_layout=True,
)
for i in range(7):
    ax3[0, 0].plot(
        simulated_deltas_sorted[-1 - i, :],
        ".-",
        label=f"deltas {bb_sorted[-1 - i]:.2E}",
    )
    ax3[0, 1].plot(simulated_deltas_sorted[i, :], ".-", label=f"BB {bb_sorted[i]:.2E}")
for axx in ax3[0, :]:
    axx.grid(ls=":")
    axx.legend()
    axx.axhline(delta_max, color="k", ls="-")

tt = np.linspace(-1, upper_int_limit, 1000)
ax3[1, 0].plot(tt, [pdf(x) for x in tt])
ax3[1, 1].hist(bb, bins=100)

for axx in ax3[1, :]:
    axx.grid(ls=":")

ax3[0, 0].set_title(f"{J} deltas resulting in 7 highest BBs")
ax3[0, 1].set_title(f"{J} deltas resulting in 7 lowest BBs")
ax3[1, 0].set_title(f"pdf")
ax3[1, 1].set_title(f"histograms of BBs")
fig3.show()

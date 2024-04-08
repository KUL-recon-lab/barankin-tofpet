# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.special import comb

from collections.abc import Callable

from pdfs import exp_pdf, biexp_pdf, truncated_gauss_pdf, truncated_double_gauss_pdf
from utils import (
    estimate_deltas,
    calculate_all_possible_U_N_ij,
    simulate_barankin_bounds,
)


# %%
# input parameters
num_possible_deltas: int = 200  # number of possible deltas
J: int = 128  # number of deltas used for one simulation
delta_min: float = 0.01  # max delta value
delta_max: float | None = 200.0  # max delta value
delta_mode: str = "log"  # delta mode: 'log' or 'lin'

num_sim: int = 500  # number of simulations

N: int = 1  # number of photons

upper_int_limit: float | None = (
    None  # upper integration limit, None means auto determined
)
rcond: float = 1e-8  # fraction of largest singular value for pinv (rcond)

# choice of the user defined pdf
pdf: Callable[[float], float] = truncated_double_gauss_pdf

# seed for random generator
seed: int = 1

np.random.seed(seed)

# %%
max_num_sim = comb(num_possible_deltas, J, exact=True)

if num_sim > max_num_sim:
    raise ValueError(
        f"Number of simulations ({num_sim}) is larger than the number of possible combinations ({max_num_sim})"
    )

# %%
# estimate deltas

print("estimating deltas")
all_possible_deltas, x0 = estimate_deltas(
    pdf,
    N,
    num_possible_deltas,
    delta_min,
    delta_max=delta_max,
    delta_mode=delta_mode,
)

if upper_int_limit is None:
    upper_int_limit = x0 + all_possible_deltas.max()
print(f"upper integration limit: {upper_int_limit:.2E}")


# plot the pdf and the upper integration limit
tt = np.linspace(-1, upper_int_limit, 1000)
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
ax.plot(tt, [pdf(x) for x in tt])
ax.set_title(f"pdf")
ax.grid(ls=":")
fig.show()

# %%
# pre-calculate all possible matrix elements U_ij using all num_possible_deltas over 2
# combinations of deltas for the integral eta(t, delta_1) eta(t, delta_2) pdf(t)

all_U_N_ij = calculate_all_possible_U_N_ij(pdf, all_possible_deltas, N, upper_int_limit)

figU, axU = plt.subplots(tight_layout=True)
imU = axU.matshow(
    all_U_N_ij,
    norm=LogNorm(vmin=all_U_N_ij.min(), vmax=all_U_N_ij.max()),
)
axU.set_title(
    f"all possible U_N_ij (delta_min={all_possible_deltas.min():.2E}, delta_max={all_possible_deltas.max():.2E})",
    fontsize="medium",
)
figU.colorbar(imU)
figU.show()


# %%

bb_sorted, simulated_deltas_sorted = simulate_barankin_bounds(
    all_U_N_ij, num_sim, J, all_possible_deltas, rcond=rcond, sort_output=True
)

print(f"highest Barankin bound (BB)          ..: {bb_sorted[-1]:.3E}")
print(f"lowest  Barankin bound (BB)          ..: {bb_sorted[0]:.3E}")
print(f"5% percentile of all Barankin bounds ..: {bb_sorted[int(0.05*num_sim)]:.3E}")

# %%
fig2, ax2 = plt.subplots(
    1,
    3,
    figsize=(12, 4),
    tight_layout=True,
)
for i in range(7):
    ax2[0].plot(
        simulated_deltas_sorted[-1 - i, :],
        ".-",
        label=f"deltas {bb_sorted[-1 - i]:.2E}",
    )
    ax2[1].plot(simulated_deltas_sorted[i, :], ".-", label=f"BB {bb_sorted[i]:.2E}")
ax2[2].hist(bb_sorted, bins=100)

for axx in ax2[:2]:
    axx.grid(ls=":")
    axx.legend()
    axx.axhline(all_possible_deltas.max(), color="k", ls="-")

ax2[2].grid(ls=":")

ax2[0].set_title(f"{J} deltas resulting in 7 highest BBs")
ax2[1].set_title(f"{J} deltas resulting in 7 lowest BBs")
ax2[2].set_title(f"histograms of BBs")

fig2.show()

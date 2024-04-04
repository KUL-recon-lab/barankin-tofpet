# TODO: - method 2 for deltas, plot of variance against new delta
# - check whether we hit max bound
# - plot deltas that yield highest bound
# - check whether choice of 2 deltas from 80 fixed values is better
# - when going to N photons make sure that power to N is stable

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.special import comb

from collections.abc import Callable


def exp_pdf(x: float, mu: float = 1.0) -> float:
    if x >= 0:
        p = mu * math.exp(-mu * x)
    else:
        p = 0
    return p


def biexp_pdf(x: float, mu1: float = 1.0, mu2: float = 0.5, a: float = 0.5) -> float:
    if x >= 0:
        p = a * mu1 * math.exp(-mu1 * x) * (1 - a) * mu2 * math.exp(-mu2 * x)
    else:
        p = 0
    return p


# %%
pdf: Callable[[float], float] = exp_pdf

# %%
shifted_pdf: Callable[[float, float], float] = lambda t, delta: pdf(t - delta)
eta: Callable[[float, float], float] = lambda t, delta: pdf(t - delta) / pdf(t) - 1

integrand: Callable[[float, float, float], float] = (
    lambda t, a, b: eta(t, a) * eta(t, b) * pdf(t)
)

# %%
# plot pdf
d1 = 1.0
d2 = 2.0

integ = lambda x: integrand(x, d1, d2)

tt = np.linspace(0, 10, 1000)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(tt, [pdf(x) for x in tt], label="pdf")
ax.plot(tt, [eta(x, d1) for x in tt], label=f"eta({d1})")
ax.plot(tt, [eta(x, d2) for x in tt], label=f"eta({d2})")
ax.plot(tt, [integ(x) for x in tt], label=f"bb integrand")
ax.grid(ls=":")
ax.legend()
fig.show()

# %%
num_sim = 200  # number of simulations
num_possible_deltas = 200  # number of possible deltas
num_deltas = 16  # number of possible deltas
delta_min = 0.01  # max delta value
delta_max = 2.0  # max delta value

rcond = 1e-8  # fraction of largest singular value for pinv (rcond)

upper_int_limit = 50.0  # upper integration limit

num_photons = 10  # number of photons

# %%

max_num_sim = comb(num_possible_deltas, num_deltas, exact=True)

if num_sim > max_num_sim:
    raise ValueError(
        f"Number of simulations ({num_sim}) is larger than the number of possible combinations ({max_num_sim})"
    )

bb = np.zeros(num_sim)

np.random.seed(1)
all_possible_deltas = np.logspace(
    np.log10(delta_min), np.log10(delta_max), num_possible_deltas
)
simulated_deltas = np.zeros((num_sim, num_deltas))

t_test = np.linspace(0, upper_int_limit, 10)

for i_sim in range(num_sim):
    deltas = np.random.choice(all_possible_deltas, size=(num_deltas,))
    deltas.sort()
    simulated_deltas[i_sim, :] = deltas

    U = np.zeros((num_deltas, num_deltas))

    for i in range(num_deltas):
        for j in range(num_deltas):
            integ = lambda x: integrand(x, deltas[i], deltas[j])

            # check whether values of integrand are reasonable
            integ_test_values = [integ(x) for x in t_test]
            if max(integ_test_values) > 1e10:
                raise ValueError(
                    f"Integrand values are too large: {max(integ_test_values)}"
                )

            if deltas[i] > deltas[j]:
                l1 = deltas[j]
                l2 = deltas[i]
            else:
                l1 = deltas[i]
                l2 = deltas[j]

            I1 = quad(integ, 0, l1)[0]
            I2 = quad(integ, l1, l2)[0]
            I3 = quad(integ, l2, upper_int_limit)[0]

            val = I1 + I2 + I3

            U[i, j] = val
            U[j, i] = val

    U_N = (U + 1) ** num_photons - 1

    pinv_U_N = np.linalg.pinv(U_N, rcond=rcond)

    bb[i_sim] = float(np.sum(deltas * (pinv_U_N @ deltas)))
    print(f"{i_sim:04}/{num_sim:04}, {bb[i_sim]:.2E}", end="\r")

print()

# %%
# sort all bounds and used deltas

i_sort = np.argsort(bb)
bb_sorted = bb[i_sort]
simulated_deltas_sorted = simulated_deltas[i_sort, :]

print(bb_sorted[-1])
print(simulated_deltas_sorted[-1, :])

# %%

# %%
fig3, ax3 = plt.subplots(
    1,
    3,
    figsize=(12, 4),
    tight_layout=True,
)
for i in range(7):
    ax3[0].plot(
        simulated_deltas_sorted[-1 - i, :],
        ".-",
        label=f"deltas {bb_sorted[-1 - i]:.2E}",
    )
    ax3[1].plot(simulated_deltas_sorted[i, :], ".-", label=f"BB {bb_sorted[i]:.2E}")
for axx in ax3[:2]:
    axx.grid(ls=":")
    axx.legend()
    axx.axhline(delta_max, color="k", ls="-")
ax3[2].hist(bb, bins=100)

ax3[0].set_title(f"deltas for highest BB")
ax3[1].set_title(f"deltas for lowest BB")
ax3[2].set_title(f"histograms of BB")
fig3.show()

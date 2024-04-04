from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad

from collections.abc import Callable


def exp_pdf(x: float) -> float:
    if x >= 0:
        p = math.exp(-x)
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

num_deltas = 16  # number of deltas
delta_max = 5.0  # max delta value

num_sim = 500  # number of simulations
rcond = 1e-6  # fraction of largest singular value for pinv (rcond)

upper_int_limit = 50.0  # upper integration limit

# %%
bb = np.zeros(num_sim)

np.random.seed(1)

for i_sim in range(num_sim):
    deltas = np.sort(delta_max * np.random.rand(num_deltas))
    U = np.zeros((num_deltas, num_deltas))

    for i in range(num_deltas):
        for j in range(num_deltas):
            integ = lambda x: integrand(x, deltas[i], deltas[j])

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

    pinv_U = np.linalg.pinv(U, rcond=rcond)

    bb[i_sim] = float(np.sum(deltas * (pinv_U @ deltas)))
    print(f"{i_sim:04}/{num_sim:04}, {bb[i_sim]:.2E}", end="\r")

print()
print(bb.max())

# %%
fig2, ax2 = plt.subplots(tight_layout=True)
ax2.hist(bb, bins=100)
fig2.show()

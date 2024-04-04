# TODO: def eta (pdf, delta)

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable
from array_api_strict._array_object import Array


def exp_pdf(x: Array) -> Array:
    if np.min(x) >= 0:
        p = np.exp(-x)
    else:
        p = np.where(x >= 0, np.exp(-x), np.zeros_like(x))
    return p


# %%
pdf: Callable[[Array], Array] = exp_pdf

# %%
shifted_pdf: Callable[[Array, float], Array] = lambda t, delta: pdf(t - delta)
eta: Callable[[Array, float], Array] = lambda t, delta: pdf(t - delta) / pdf(t) - 1

# %%
t = np.linspace(0, 50, 10000, endpoint=False)
dt = float(t[1] - t[0])
p = pdf(t)

num_deltas = 16
num_sim = 100
bb = np.zeros(num_sim)

np.random.seed(1)

for i_sim in range(num_sim):
    deltas = np.sort(10 * np.random.rand(num_deltas))
    U = np.zeros((num_deltas, num_deltas))

    for i in range(num_deltas):
        eta_i = eta(t, float(deltas[i]))
        tmp = eta_i * p
        for j in range(num_deltas):
            eta_j = eta(t, float(deltas[j]))
            U[i, j] = np.sum(eta_j * tmp) * dt
            U[j, i] = U[i, j]

    pinv_U = np.linalg.pinv(U, rcond=1e-6)

    bb[i_sim] = float(np.sum(deltas * (pinv_U @ deltas)))
    print(f"{i_sim:04}/{num_sim:04}, {bb[i_sim]:.2E}", end="\r")

print()

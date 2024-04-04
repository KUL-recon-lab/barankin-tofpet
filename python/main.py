# TODO: def eta (pdf, delta)

from __future__ import annotations

import array_api_strict as xp
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable
from array_api_strict._array_object import Array


def exp_pdf(x: Array) -> Array:
    if xp.min(x) >= 0:
        p = xp.exp(-x)
    else:
        p = xp.where(x >= 0, xp.exp(-x), xp.zeros_like(x))
    return p


# %%
pdf: Callable[[Array], Array] = exp_pdf

# %%
shifted_pdf: Callable[[Array, float], Array] = lambda t, delta: pdf(t - delta)
eta: Callable[[Array, float], Array] = lambda t, delta: pdf(t - delta) / pdf(t) - 1

# %%
t = xp.linspace(0, 50, 10000, endpoint=False)
dt = float(t[1] - t[0])
p = pdf(t)

num_deltas = 16
num_sim = 2000
bb = xp.zeros(num_sim)

np.random.seed(1)

for i_sim in range(num_sim):
    deltas = xp.sort(xp.asarray(10 * np.random.rand(num_deltas)))
    U = xp.zeros((num_deltas, num_deltas))

    for i in range(num_deltas):
        eta_i = eta(t, float(deltas[i]))
        tmp = eta_i * p
        for j in range(num_deltas):
            eta_j = eta(t, float(deltas[j]))
            U[i, j] = xp.sum(eta_j * tmp) * dt
            U[j, i] = U[i, j]

    pinv_U = xp.linalg.pinv(U, rtol=1e-6)

    bb[i_sim] = float(xp.sum(deltas * (pinv_U @ deltas)))
    print(i_sim, bb[i_sim])

# fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
# for i, axx in enumerate(ax.ravel()):
#    d = 0.2 * i + 0.5
#    axx.plot(np.asarray(t), np.asarray(pdf(t)))
#    axx.plot(np.asarray(t), np.asarray(shifted_pdf(t, d)))
#    axx.plot(np.asarray(t), np.asarray(eta(t, d)) * pdf(t))
# fig.show()

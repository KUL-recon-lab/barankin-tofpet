# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.integrate import quad

from collections.abc import Callable

from pdfs import det2_pdf
from utils import barankin_bound


# %%
# input parameters
num_possible_deltas: int = 80  # number of possible deltas
delta_min: float | None = None  # max delta value
delta_max: float | None = None  # max delta value
delta_mode: str = "log"  # delta mode: 'log' or 'lin'
N: int = 10  # number of photons
Jmax: int = 32
x_zero: float | None = None  # point beyond which pdf is essentially zero
rcond: float = 1e-12  # fraction of largest singular value for pinv (rcond)

# choice of the user defined pdf
pdf: Callable[[float], float] = det2_pdf

# show interactive plots
interactive: bool = False

# %%
# estimate delta_min and delta_max
if x_zero is None:
    xx = np.logspace(-8, 8, 1000)
    y = np.array([pdf(x) for x in xx])
    x_zero = xx[np.where(y >= 1e-5)[0].max() + 1]

# normalize the pdf
norm = quad(pdf, 0, x_zero)[0]
normalized_pdf = lambda x: pdf(x) / norm

stddev_pdf = np.sqrt(
    quad(lambda x: (x**2) * normalized_pdf(x), 0, x_zero)[0]
    - quad(lambda x: x * normalized_pdf(x), 0, x_zero)[0] ** 2
)

if delta_max is None:
    delta_max = 100.0 * stddev_pdf / N
if delta_min is None:
    delta_min = 0.001 * stddev_pdf / N

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


# %%
upper_int_limit = x_zero + delta_max

# plot the pdf and the upper integration limit
tt = np.concatenate(([-1], np.linspace(0, x_zero, 1000)))
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
ax.plot(tt, [normalized_pdf(x) for x in tt])
ax.set_title(f"normalized pdf")
ax.grid(ls=":")
fig.show()

# %%
# estimate the Barankin bound for the varinace
bbs, chosen_deltas = barankin_bound(
    normalized_pdf,
    all_possible_deltas,
    N,
    Jmax,
    upper_int_limit,
    rcond=rcond,
    interactive=interactive,
)

# %%
# visualize the results

Js = np.arange(1, Jmax + 1)

fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax3[0].plot(Js, np.sqrt(bbs), ".-")
ax3[1].semilogy(Js, chosen_deltas, ".-")
ax3[2].semilogy(Js, sorted(chosen_deltas), ".-")
for axx in ax3:
    axx.grid(ls=":")
    axx.set_xlabel("J")
for axx in ax3[1:]:
    axx.axhline(delta_min, color="k", ls="--")
    axx.axhline(delta_max, color="k", ls="--")
ax3[0].set_title("barankin bound for std.dev.")
ax3[1].set_title("chosen deltas")
ax3[2].set_title("sorted chosen deltas")
fig3.show()

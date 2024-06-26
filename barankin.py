# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.interpolate import interp1d

from collections.abc import Callable
from pdfs import det2_pdf
from utils import barankin_bound


# %%
# input parameters

# number of possible deltas
num_possible_deltas: int = 256
# minimum delta to consider, None mean auto determined
delta_min: float | None = None
# maximum delta to consider, None mean auto determined
delta_max: float | None = None
# number of photons / samples
N: int = 100
# maximum J value
Jmax: int = 32
# point beyond which pdf is essentially zero, None means auto determined
x_zero: float | None = None
# fraction of largest singular value for calculate of pseudo inverse
rcond: float = 1e-12
# show interactive plots on how J values are chose, requires user interaction
interactive: bool = False
# show condition number of U matrix
show_cond_number: bool = False


# %%

pdf: Callable[[float], float] = det2_pdf

# %%
# estimate the point beyond which the pdf is essentially zero (smaller than 1e-5)

if x_zero is None:
    xx = np.logspace(-8, 8, 1000)
    test_pdf = np.array([pdf(x) for x in xx])
    x_zero = xx[np.where(test_pdf >= 1e-5)[0].max() + 1]

# %%
# normalize the pdf

norm = quad(pdf, 0, x_zero)[0]
normalized_pdf = lambda x: pdf(x) / norm

stddev_pdf = np.sqrt(
    quad(lambda x: (x**2) * normalized_pdf(x), 0, x_zero)[0]
    - quad(lambda x: x * normalized_pdf(x), 0, x_zero)[0] ** 2
)

# %%
# plot the pdf
iplot_max = np.where(test_pdf >= 0.05 * test_pdf.max())[0].max() + 1

fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True, sharex=True)
ax[0].plot(xx[:iplot_max], test_pdf[:iplot_max])
ax[0].set_title(r"input pdf $\tilde{p}(x)$")
ax[0].grid(ls=":")
ax[0].set_xlabel("x")
ax[1].plot(xx[:iplot_max], test_pdf[:iplot_max] / norm)
ax[1].set_title(r"normalized pdf $p(x)$ with $\int_0^\infty p(x) dx = 1$")
ax[1].grid(ls=":")
ax[1].set_xlabel("x")
fig.show()

# %%
# estimate the min / max delta value to be considered

if delta_max is None:
    delta_max = 100.0 * stddev_pdf / N
if delta_min is None:
    delta_min = 0.001 * stddev_pdf / N

print(f"delta_min: {delta_min:.2E}")
print(f"delta_max: {delta_max:.2E}")

# setup the geometric deltas series
all_possible_deltas: list[float] = np.logspace(
    np.log10(delta_min), np.log10(delta_max), num_possible_deltas
)

upper_int_limit = x_zero + delta_max

# %%
# estimate the Barankin bound for the varinace
bbs, chosen_deltas, U = barankin_bound(
    normalized_pdf,
    all_possible_deltas,
    N,
    Jmax,
    upper_int_limit,
    rcond=rcond,
    interactive=interactive,
    show_cond_number=show_cond_number,
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
ax3[0].set_title(f"barankin bound for std.dev. and N={N}")
ax3[1].set_title("chosen deltas")
ax3[2].set_title("sorted chosen deltas")
fig3.show()

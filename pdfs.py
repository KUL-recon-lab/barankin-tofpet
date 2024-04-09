"""example pdfs for testing the Barankin bound"""

import math


def exp_pdf(x: float, mu: float = 1.0) -> float:
    """exponential pdf

    Parameters
    ----------
    x : float
        independent variable
    mu : float, optional
        exponent, by default 1.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = mu * math.exp(-mu * x)
    else:
        p = 0
    return p


def truncated_gauss_pdf(x: float, mu: float = 2.0, sig: float = 3.0) -> float:
    """truncate gaussian pdf (0 for x < 0)

    Parameters
    ----------
    x : float
        independent variable
    mu : float, optional
        mean, default 2.0
    sig : float, optional
        standard deviation, default 3.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = math.exp(-0.5 * ((x - mu) / sig) ** 2)
    else:
        p = 0
    return p


def truncated_double_gauss_pdf(
    x: float,
    mu1: float = 7.0 / 100,
    sig1: float = 2.0 / 100,
    mu2: float = 150.0 / 100,
    sig2: float = 60.0 / 100,
) -> float:
    """truncate double gaussian pdf (0 for x < 0)

    Parameters
    ----------
    x : float
        independent variable
    mu1/mu2 : float, optional
        mean, default 7.0/70.9
    sig1/sig2 : float, optional
        standard deviation, default 3.0/60.0

    Returns
    -------
    float
    """
    if x >= 0:
        p = (0.1 / math.sqrt(2 * math.pi * sig1**2)) * math.exp(
            -0.5 * ((x - mu1) / sig1) ** 2
        ) + (0.9 / math.sqrt(2 * math.pi * sig2**2)) * math.exp(
            -0.5 * ((x - mu2) / sig2) ** 2
        )
    else:
        p = 0
    return p


def _a(t: float, tau: float, t_tr: float, sig_tr: float) -> float:
    A = 0.5 * math.exp(-((t - t_tr) / tau) + 0.5 * sig_tr**2 / tau**2)
    B = math.erfc((t_tr + sig_tr**2 / tau - t) / (math.sqrt(2) * sig_tr))
    C = math.erfc((t_tr + sig_tr**2 / tau) / (math.sqrt(2) * sig_tr))

    return A * (B - C)


def bi_exp_model(
    t: float, tau_d: float, tau_r: float, t_tr: float, sig_tr: float
) -> float:

    if t < 0:
        return 0

    else:
        C = 2 / (1 + math.erfc(t_tr / (math.sqrt(2) * sig_tr)))
        return (C / (tau_d - tau_r)) * (
            _a(t, tau_d, t_tr, sig_tr) - _a(t, tau_r, t_tr, sig_tr)
        )


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    tt = np.linspace(0, 30, 10000)

    p1 = np.array([bi_exp_model(t, 1.0, 0.01, 0.179, 0.081) for t in tt])
    p2 = np.array([bi_exp_model(t, 20.0, 5.0, 0.179, 0.081) for t in tt])

    alpha = 0.05
    p = alpha * p1 + (1 - alpha) * p2

    plt.plot(tt, p, ".-")
    plt.show()

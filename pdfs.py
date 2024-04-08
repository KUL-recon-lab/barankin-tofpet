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
    sig1: float = 3.0 / 100,
    mu2: float = 70.0 / 100,
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
        p = math.exp(-0.5 * ((x - mu1) / sig1) ** 2) + 0.3 * math.exp(
            -0.5 * ((x - mu2) / sig2) ** 2
        )
    else:
        p = 0
    return p


def biexp_pdf(x: float, mu1: float = 3.0, mu2: float = 0.5, a: float = 0.5) -> float:
    """bi-exponential pdf

    Parameters
    ----------
    x : float
        independent variable
    mu1/mu2 : float, optional
        exponent, by default 3.0 / 0.5
    a : float, optional
        weight of the first exponential is a
        weight of the second exponential is (1-a), by default 0.5

    Returns
    -------
    float
    """
    if x >= 0:
        p = a * mu1 * math.exp(-mu1 * x) * (1 - a) * mu2 * math.exp(-mu2 * x)
    else:
        p = 0
    return p

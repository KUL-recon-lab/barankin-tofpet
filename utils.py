from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


def estimate_deltas(
    pdf: Callable[[float], float],
    N: int,
    num_possible_deltas: int,
    delta_min: float,
    delta_max: float | None = None,
    delta_mode: str = "log",
) -> tuple[np.ndarray, float]:
    """heuristic estimation of deltas for Barankin bound calculation

    Parameters
    ----------
    pdf : Callable[[float], float]
        probability density function
    N : int
        number of samples / photons
    num_possible_deltas : int
        number of possible deltas to consider
    delta_min : float
        minimum delta
    delta_max : float | None, optional
        maximum delta, if None, it will be estimated,
        by default None
    delta_mode : str, optional
        arange deltas in linspace or logspace, by default "log"

    Returns
    -------
    tuple[np.ndarray, float]
        all_possible_deltas, point beyond which pdf is essentially zero
    """

    eta: Callable[[float, float], float] = lambda t, delta: pdf(t - delta) / pdf(t) - 1
    integrand: Callable[[float, float, float], float] = (
        lambda t, a, b: eta(t, a) * eta(t, b) * pdf(t)
    )

    # smallest x where pdf(x) is below 1e-8 which we use as upper integration limit
    x_zero = root_scalar(lambda x: pdf(x) - 1e-8, x0=10, bracket=[0, 1e6]).root

    # check the largest and smallest possible U_ij
    integ_min = lambda x: integrand(x, delta_min, delta_min)
    Umin = (
        quad(integ_min, 0, delta_min)[0]
        + quad(integ_min, delta_min, x_zero + delta_min)[0]
    )
    U_N_min = (Umin + 1) ** N - 1

    if delta_max is None:
        delta_max = 1.01 * delta_min
        it = 0
        for it in range(100):
            integ_max = lambda x: integrand(x, delta_max, delta_max)
            Umax = (
                quad(integ_max, 0, delta_max)[0]
                + quad(integ_max, delta_max, x_zero + delta_max)[0]
            )
            U_N_max = (Umax + 1) ** N - 1
            log_dyn_range = np.log10(U_N_max / U_N_min)
            print(f"{it:03} {log_dyn_range:.2f}", end="\r")

            if log_dyn_range < 7.0:
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
        U_N_max = (Umax + 1) ** N - 1
        log_dyn_range = np.log10(U_N_max / U_N_min)

    print(f"delta min / pdf(delta min): {delta_min:.2E} / {pdf(delta_min):.2E}")
    print(f"delta max / pdf(delta max): {delta_max:.2E} / {pdf(delta_max):.2E}")
    print(f"log dynamic range of all possible U_N_ij: {log_dyn_range:.2f}")

    if log_dyn_range > 15.0:
        raise ValueError("Dynamic range of U_N_ij is too large. Decrease delta_max.")
    if log_dyn_range < 7.0:
        raise ValueError("Dynamic range of U_N_ij is too low. Increase delta_max.")

    if delta_mode == "log":
        all_possible_deltas = np.logspace(
            np.log10(delta_min), np.log10(delta_max), num_possible_deltas
        )
    elif delta_mode == "lin":
        all_possible_deltas = np.linspace(delta_min, delta_max, num_possible_deltas)
    else:
        raise ValueError("delta_mode must be 'log' or 'lin'")

    return all_possible_deltas, x_zero


def calculate_all_possible_U_N_ij(
    pdf: Callable[[float], float],
    all_possible_deltas: np.ndarray,
    N: int,
    upper_int_limit: float,
) -> np.ndarray:
    """calculate all matrix elements U_N_ij for all possible deltas

    Parameters
    ----------
    pdf : Callable[[float], float]
        probability density function
    all_possible_deltas : np.ndarray
        all deltas to consider
    N : int
        number of samples / photons
    upper_int_limit : float
        upper integration limit

    Returns
    -------
    np.ndarray
        A 2D array containing all possible U_N_ij
    """

    num_possible_deltas = all_possible_deltas.shape[0]

    eta: Callable[[float, float], float] = lambda t, delta: pdf(t - delta) / pdf(t) - 1
    integrand: Callable[[float, float, float], float] = (
        lambda t, a, b: eta(t, a) * eta(t, b) * pdf(t)
    )
    print(
        f"pre-calculate all possible ({num_possible_deltas*(num_possible_deltas-1)//2}) matrix elements U_ij"
    )

    all_U_ij = np.zeros((num_possible_deltas, num_possible_deltas))

    for i in range(num_possible_deltas):
        for j in range(i + 1):
            integ = lambda x: integrand(
                x, all_possible_deltas[i], all_possible_deltas[j]
            )
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

    all_U_N_ij = (all_U_ij + 1) ** N - 1

    return all_U_N_ij


def simulate_barankin_bounds(
    all_U_N_ij: np.ndarray,
    num_sim: int,
    J: int,
    all_possible_deltas: np.ndarray,
    rcond: float = 1e-8,
    sort_output: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """simulate Barankin bounds based on random sub samples of all possible deltas

    Parameters
    ----------
    all_U_N_ij : np.ndarray
        2D array with pre-calculated U_N_ij
    num_sim : int
        number of simulations to perform
    J : int
        number of deltas to choose for each simulation
    all_possible_deltas : np.ndarray
        all possible deltas to consider
    rcond : float, optional
        passed to np.linalg.pinv, by default 1e-8
    sort_output : bool, optional
        sort output array by ascending bounds, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        simulated Barankin bounds and used deltas
    """
    bb = np.zeros(num_sim)
    simulated_deltas = np.zeros((num_sim, J))

    all_inds = np.arange(all_possible_deltas.shape[0])

    for i_sim in range(num_sim):
        inds = np.random.choice(all_inds, size=(J,))
        inds.sort()

        deltas = all_possible_deltas[inds]
        simulated_deltas[i_sim, :] = deltas

        U_N = np.zeros((J, J))

        for i, ii in enumerate(inds):
            for j, jj in enumerate(inds):
                U_N[i, j] = all_U_N_ij[ii, jj]

        pinv_U_N = np.linalg.pinv(U_N, rcond=rcond)

        bb[i_sim] = float(np.sum(deltas * (pinv_U_N @ deltas)))
        print(f"{(i_sim+1):04}/{num_sim:04}, {bb[i_sim]:.3E}", end="\r")

    print()

    if sort_output:
        i_sort = np.argsort(bb)
        bb_sorted = bb[i_sort]
        simulated_deltas_sorted = simulated_deltas[i_sort, :]
    else:
        bb_sorted = bb
        simulated_deltas_sorted = simulated_deltas

    return bb_sorted, simulated_deltas_sorted


def U_N_ij(
    pdf: Callable[[float], float],
    delta_i: float,
    delta_j: float,
    N: int,
    upper_int_limit: float,
) -> float:
    """calculate matrix elements U_N_ij

    Parameters
    ----------
    pdf : Callable[[float], float]
        probability density function
    delta_i : float
        first delta shift
    delta_j : float
        second delta shift
    N : int
        number of samples / photons
    upper_int_limit : float
        upper integration limit

    Returns
    -------
    float
        the matrix element U_N_ij
    """

    eta: Callable[[float, float], float] = lambda t, delta: pdf(t - delta) / pdf(t) - 1
    integ: Callable[[float], float] = (
        lambda t: eta(t, delta_i) * eta(t, delta_j) * pdf(t)
    )

    l1 = min(delta_i, delta_j)
    l2 = max(delta_i, delta_j)

    if l1 != l2:
        IL = quad(integ, 0, l1)
        IM = quad(integ, l1, l2)
        IR = quad(integ, l2, upper_int_limit)
        val = IL[0] + IM[0] + IR[0]
    else:
        IL = quad(integ, 0, l1)
        IR = quad(integ, l1, upper_int_limit)
        val = IL[0] + IR[0]

    return (val + 1) ** N - 1

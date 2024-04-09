import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
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
        lm = l2 + 0.01 * (upper_int_limit - l2)
        IL = quad(integ, 0, l1)
        IM = quad(integ, l1, l2)
        IR1 = quad(integ, l2, lm)
        IR2 = quad(integ, lm, upper_int_limit)
        val = IL[0] + IM[0] + IR1[0] + IR2[0]
    else:
        lm = l1 + 0.01 * (upper_int_limit - l1)
        IL = quad(integ, 0, l1)
        IM = quad(integ, l1, lm)
        IR = quad(integ, lm, upper_int_limit)
        val = IL[0] + IM[0] + IR[0]

    return (val + 1) ** N - 1


def barankin_bound(
    normalized_pdf: Callable[[float], float],
    all_possible_deltas: list[float],
    N: int,
    Jmax: int,
    upper_int_limit: float,
    rcond: float = 1e-8,
    interactive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    available_delta_inds = np.arange(all_possible_deltas.size).tolist()
    chosen_delta_inds = []

    test_bbs = np.zeros(all_possible_deltas.size)
    U_Ns = []

    U_N_ij_lut = dict()

    for j, delta in enumerate(all_possible_deltas):
        test_deltas = np.array([delta])
        U_N = np.array([[U_N_ij(normalized_pdf, delta, delta, N, upper_int_limit)]])
        U_N_ij_lut[(j, j)] = U_N[0, 0]
        U_Ns.append(U_N)

        test_bbs[j] = test_deltas.T @ (np.linalg.pinv(U_N, rcond=rcond) @ test_deltas)

    # picks deltas step by step starting from J=1 case
    i_delta_max = np.argmax(test_bbs)
    if i_delta_max == (all_possible_deltas.size - 1):
        warnings.warn("Hit upper bound of deltas. Increase upper bound and rerun.")
    elif i_delta_max == 0:
        warnings.warn("Hit lower bound of deltas. Decrease lower bound and rerun.")

    U_cur = U_Ns[i_delta_max]
    chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))

    bbs = [test_bbs[i_delta_max]]

    for J in range(Jmax - 1):
        nd = len(chosen_delta_inds)
        U_Ns = []
        test_bbs = np.zeros(len(available_delta_inds))

        for i_j, j in enumerate(available_delta_inds):
            delta = all_possible_deltas[j]
            U_N = np.zeros((nd + 1, nd + 1))
            U_N[:nd, :nd] = U_cur

            for k, i_del in enumerate(chosen_delta_inds):
                i1 = min(j, i_del)
                i2 = max(j, i_del)

                if (i1, i2) in U_N_ij_lut:
                    U_N[k, -1] = U_N_ij_lut[(i1, i2)]
                else:
                    U_N[k, -1] = U_N_ij(
                        normalized_pdf,
                        all_possible_deltas[i_del],
                        delta,
                        N,
                        upper_int_limit,
                    )
                    U_N_ij_lut[(i1, i2)] = U_N[k, -1]

                U_N[-1, k] = U_N[k, -1]

            U_N[-1, -1] = U_N_ij_lut[(j, j)]

            U_Ns.append(U_N)
            test_deltas = all_possible_deltas[np.concatenate((chosen_delta_inds, [j]))]
            test_bbs[i_j] = test_deltas.T @ (
                np.linalg.pinv(U_N, rcond=rcond) @ test_deltas
            )

        i_delta_max = np.argmax(test_bbs)

        if available_delta_inds[i_delta_max] == (all_possible_deltas.size - 1):
            warnings.warn("Hit upper bound of deltas. Increase upper bound and rerun.")
        elif available_delta_inds[i_delta_max] == 0:
            warnings.warn("Hit lower bound of deltas. Decrease lower bound and rerun.")

        if interactive:
            figi, axi = plt.subplots(figsize=(8, 4), tight_layout=True)
            axi.plot(available_delta_inds, test_bbs)
            axi.axvline(available_delta_inds[i_delta_max], color="r", ls="--")
            figi.show()
            tmp = input(">")
            plt.close()

        chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))
        U_cur = U_Ns[i_delta_max]
        bbs.append(test_bbs[i_delta_max])
        print(
            f"{(J + 2):04}  {test_bbs[i_delta_max]:.4E} {np.sqrt(test_bbs[i_delta_max]):.4E}",
            end="\r",
        )

    print()

    return np.array(bbs), np.array(all_possible_deltas[chosen_delta_inds])

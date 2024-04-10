import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from scipy.integrate import quad


def eta(t: float, delta: float, pdf: Callable[[float], float]):
    A = pdf(t - delta)
    B = pdf(t)

    if A == 0 and B == 0:
        return 0
    elif A != 0 and B == 0:
        return np.finfo(np.float32).max
    else:
        return pdf(t - delta) / pdf(t) - 1


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

    integ: Callable[[float], float] = (
        lambda t: eta(t, delta_i, pdf) * eta(t, delta_j, pdf) * pdf(t)
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
    verbose: bool = True,
    show_cond_number: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    normalized_pdf : Callable[[float], float]
        normalized probability density function which must return 0 for x < 0
    all_possible_deltas : list[float]
        list of deltas to consider to calculate Barankin bound
    N : int
        number of photons / samples
    Jmax : int
        maximum J (number of deltas) to consider
    upper_int_limit : float
        upper limit for evaluation of integrals
    rcond : float, optional
        see np.linalg.pinv, by default 1e-8
    interactive : bool, optional
        show interactive plots showing choice of next delta, by default False
    verbose : bool, optional
        print verbose output, by default True
    show_cond_number : bool, optional
        show condition number of U matrix, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Barankin bound for variance as function of J
        chosen delta values
        U matrix
    """

    if show_cond_number and not verbose:
        verbose = True

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

        test_bbs[j] = test_deltas.T @ (
            np.linalg.pinv(U_N, rcond=rcond, hermitian=True) @ test_deltas
        )

    # picks deltas step by step starting from J=1 case
    i_delta_max = np.argmax(test_bbs)
    if i_delta_max == (all_possible_deltas.size - 1):
        warnings.warn("Hit upper bound of deltas. Increase upper bound and rerun.")
    elif i_delta_max == 0:
        warnings.warn("Hit lower bound of deltas. Decrease lower bound and rerun.")

    U_cur = U_Ns[i_delta_max]
    chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))

    bbs = [test_bbs[i_delta_max]]

    for J in np.arange(2, Jmax + 1):
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
                np.linalg.pinv(U_N, rcond=rcond, hermitian=True) @ test_deltas
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
            _ = input(">")
            plt.close()

        chosen_delta_inds.append(available_delta_inds.pop(i_delta_max))
        U_cur = U_Ns[i_delta_max]
        bbs.append(test_bbs[i_delta_max])
        if verbose:
            if show_cond_number:
                _, S, _ = np.linalg.svd(U_cur, hermitian=True)
                cond_num = S.max() / S.min()
                if cond_num > 1 / rcond:
                    warnings.warn("Large condition number for matrix U.")
                cond_str = f" U C-NUM: {cond_num:.2E}"
            else:
                cond_str = ""
            print(
                f"J: {J:04} BB-VAR: {test_bbs[i_delta_max]:.4E} BB-STDDEV: {np.sqrt(test_bbs[i_delta_max]):.4E}{cond_str}",
                end="\r",
            )

    if verbose:
        print()

    return np.array(bbs), np.array(all_possible_deltas[chosen_delta_inds]), U_cur

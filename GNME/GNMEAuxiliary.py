r"""
Code based off "Generalized nonorthogonal matrix elements: Unifying Wick's theorem and the Slater--Condon rules"
"""
import numpy as np

THRESH=1e-8     # Threshold for an eigenvalue to be considered zero


def lowdin_pairing(docc, sao, cocc):
    r"""
    Bi-orthogonalise orbitals
    :param docc: *Occupied* MO coefficients for bra state
    :param sao: AO overlap matrix
    :param cocc: *Occupied* MO coefficients for ket state
    :return:
    """
    overlap = np.einsum("ip,ij,jq->pq", docc, sao, cocc, optimize="optimal")
    U, s, Vh = np.linalg.svd(overlap)
    dtrf = np.dot(docc, U)
    ctrf = np.dot(cocc, Vh.T)
    dtrf[:, 0] *= np.linalg.det(U)
    ctrf[:, 0] *= np.linalg.det(Vh)
    return dtrf, ctrf, s


def get_unweighted_codensity(dtrf, ctrf, s):
    r"""
    Get the unweighted codensity matrix P (Equation 38b)
    :param dtrf:
    :param ctrf:
    :param s:
    :return:
    """
    n = len([x for x in s if x > THRESH])
    coden = np.einsum("ik,jk->ij", dtrf[:, n:], ctrf[:, n:])
    return coden


def get_weighted_codensity(dtrf, ctrf, s):
    r"""
    Get the weighted codensity matrix W (Equation 38c)
    :param dtrf:
    :param ctrf:
    :param s:
    :return:
    """
    n = len([x for x in s if x > THRESH])
    wcoden = np.einsum("ik,k,jk->ij", dtrf[:, :n], 1/s[:n], ctrf[:, :n])
    return wcoden


def get_parity(arr):
    r"""
    Get parity of permutation via bubble sort
    :param arr:
    :return:
    """
    n = len(arr)
    swapped = False
    perm = 0  # Number of permutations
    # Traverse through all array elements
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                perm += 1
        if not swapped:
            break
    if perm % 2 == 0:
        return 1
    else:
        return -1

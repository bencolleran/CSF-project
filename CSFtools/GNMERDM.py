r"""
Get RDM with GNME
"""
import numpy as np
import string
from itertools import permutations
from scipy.linalg import block_diag
from CSFtools.GNMEAuxiliary import lowdin_pairing, get_weighted_codensity, get_unweighted_codensity, get_parity

THRESH = 1e-8  # Threshold for an eigenvalue to be considered zero


def antisymmetrise_contraction(fcontraction):
    r"""

    :param fcontraction:
    :return:
    """
    n = len(fcontraction.shape) // 2
    total_contraction = np.zeros(fcontraction.shape)

    orig_str = string.ascii_lowercase[15:15 + 2 * n]
    perm = permutations(orig_str[n:])
    for i in list(perm):
        new_str = ''.join(orig_str[:n].split() + list(i))
        pf = get_parity([*new_str])
        total_contraction += np.einsum(f"{orig_str}->{new_str}", fcontraction) * pf
    return total_contraction


def get_one_rdm_contrib(ddet, dmo, cdet, cmo, sao, ao2mo):
    r"""
    Get contributions to the 1-RDM for two non-orthogonal determinants (ddet and cdet), each with their MO coefficients
    (dmo, cmo). 'd' labels the bra state and 'c' labels the ket state
    :param ddet:
    :param dmo:
    :param cdet:
    :param cmo:
    :param sao:
    :return:
    """
    nmos = ao2mo.shape[0]

    # Biorthogonalise orbitals
    dtrf, ctrf, s = lowdin_pairing(dmo[:, [x for x, occno in enumerate(ddet[1:]) if occno == 1]],
                                   sao,
                                   cmo[:, [x for x, occno in enumerate(cdet[1:]) if occno == 1]])

    # Find number of zeroes
    m = len([x for x in s if x < THRESH])

    # Find reduced overlap
    redS = np.prod([x for x in s if x > THRESH])

    M = get_unweighted_codensity(dtrf, dtrf, s) + get_unweighted_codensity(dtrf, ctrf, s) + \
        get_weighted_codensity(dtrf, ctrf, s)
    P = get_unweighted_codensity(dtrf, ctrf, s)

    # Transform to active MO basis
    M = np.einsum("kl,pk,ql->pq", M, ao2mo, ao2mo, optimize="optimal")
    P = np.einsum("kl,pk,ql->pq", P, ao2mo, ao2mo, optimize="optimal")

    # Return zero matrix if m > 1
    if m > 1:
        return np.zeros((nmos, nmos))
    # Or return non-zero matrix if m <= 1
    elif m == 0:
        return M * redS
    else:  # m == 1
        return P * redS


def get_two_rdm_contrib(ddet, dmo, cdet, cmo, sao, ao2mo):
    r"""
    Get contributions to the 2-RDM for two non-orthogonal determinants (ddet and cdet), each with their MO coefficients
    (dmo, cmo). 'd' labels the bra state and 'c' labels the ket state
    :param ddet:
    :param dmo:
    :param cdet:
    :param cmo:
    :param sao:
    :return:
    """
    nmos = ao2mo.shape[0]

    # Biorthogonalise orbitals
    dtrf, ctrf, s = lowdin_pairing(dmo[:, [x for x, occno in enumerate(ddet[1:]) if occno == 1]],
                                   sao,
                                   cmo[:, [x for x, occno in enumerate(cdet[1:]) if occno == 1]])

    # Find number of zeroes
    m = len([x for x in s if x < THRESH])

    # Find reduced overlap
    redS = np.prod([x for x in s if x > THRESH])

    M = get_unweighted_codensity(dtrf, dtrf, s) + get_unweighted_codensity(dtrf, ctrf, s) + \
        get_weighted_codensity(dtrf, ctrf, s)
    P = get_unweighted_codensity(dtrf, ctrf, s)

    # Transform to active MO basis
    M = np.einsum("kl,pk,ql->pq", M, ao2mo, ao2mo, optimize="optimal")
    P = np.einsum("kl,pk,ql->pq", P, ao2mo, ao2mo, optimize="optimal")

    # Return zero matrix if m > 2
    if m > 2:
        return np.zeros((nmos, nmos, nmos, nmos))
    elif m == 0:  # There is 1 way of partitioning the (lack of) zeroes
        fcontraction = np.einsum("rp,sq->pqrs", M, M, optimize='optimal')  # Define fundamental contraction
        return antisymmetrise_contraction(fcontraction) * redS
    elif m == 1:  # There are 3 ways of partitioning "1"
        fcontraction = np.einsum("rp,sq->pqrs", M, P, optimize='optimal') + \
                       np.einsum("rp,sq->pqrs", P, M, optimize='optimal')
        return antisymmetrise_contraction(fcontraction) * redS
    else:  # m == 2
        fcontraction = np.einsum("rp,sq->pqrs", P, P, optimize='optimal')  # Define fundamental contraction
        return antisymmetrise_contraction(fcontraction) * redS


def get_gnme_rdm(bras, kets, bra_coeffs, ket_coeffs, mos, ref_mo, sao, degree):
    r"""
    Get RDM from GNME code
    :param ddets:
    :param dcoeffs:
    :param dmos:
    :param cdets:
    :param ccoeffs:
    :param cmos:
    :param sao:
    :param degree:
    :return:
    """
    if len(mos) == 1:
        assert len(kets) == len(ket_coeffs)
        n = len(kets)
        m = len(bras)
        ndims = ref_mo.shape[1]

        # Generate ao2mo
        ao2mo = np.einsum("ip,iq->pq", ref_mo, sao)

        if degree == 1:
            rdm = np.zeros((ndims, ndims))
            for i in range(n):
                for j in range(n):
                    rdm += get_one_rdm_contrib(kets[i], mos[0], bras[j], mos[0], sao, ao2mo) * ket_coeffs[i] * bra_coeffs[j] * \
                           kets[i][0] * bras[j][0]
            return rdm

        if degree == 2:
            rdm = np.zeros((ndims, ndims, ndims, ndims))
            for i in range(n):
                for j in range(n):
                    #print(f"i: {i}; j: {j}")
                    rdm += get_two_rdm_contrib(kets[i], mos[0], bras[j], mos[0], sao, ao2mo) * ket_coeffs[i] * bra_coeffs[j] * \
                           kets[i][0] * bras[j][0]
            return rdm
    '''else:
        assert len(dets) == len(coeffs) == len(mos)
        n = len(dets)
        ndims = ref_mo.shape[1]

        # Generate ao2mo
        ao2mo = np.einsum("ip,iq->pq", ref_mo, sao)

        if degree == 1:
            rdm = np.zeros((ndims, ndims))
            for i in range(n):
                for j in range(n):
                    rdm += get_one_rdm_contrib(dets[i], mos[i], dets[j], mos[j], sao, ao2mo) * coeffs[i] * coeffs[j] * dets[i][0] * dets[j][0]
            return rdm

        if degree == 2:
            rdm = np.zeros((ndims, ndims, ndims, ndims))
            for i in range(n):
                for j in range(n):
                    rdm += get_two_rdm_contrib(dets[i], mos[i], dets[j], mos[j], sao, ao2mo) * coeffs[i] * coeffs[j] * dets[i][0] * dets[j][0]
            return rdm'''



# def get_one_rdm(ddets, dcoeffs, dmos, cdets, ccoeffs, cmos, sao):
#     r"""
#     Construct 1-RDM in AO basis between two arbitrary linear combination of determinants
#     :param ddets:
#     :param dcoeffs:
#     :param dmos:
#     :param cdets:
#     :param ccoeffs:
#     :param cmos:
#     :return:
#     """
#     assert len(ddets) == len(dcoeffs) == len(dmos) == len(cdets) == len(ccoeffs) == len(cmos)
#     n = len(ddets)
#     naos = dmos[0].shape[0]
#
#     rdm = np.zeros((naos, naos))
#     for i in range(n):
#         for j in range(n):
#             rdm += get_one_rdm_contrib(ddets[i], dmos[i], cdets[j], cmos[j], sao)
#     return rdm

# def get_two_rdm(ddets, dcoeffs, dmos, cdets, ccoeffs, cmos, sao):
#     r"""
#     Construct 2-RDM in AO basis between two arbitrary linear combination of determinants
#     :param ddets:
#     :param dcoeffs:
#     :param dmos:
#     :param cdets:
#     :param ccoeffs:
#     :param cmos:
#     :return:
#     """
#     assert len(ddets) == len(dcoeffs) == len(dmos) == len(cdets) == len(ccoeffs) == len(cmos)
#     n = len(ddets)
#     naos = dmos[0].shape[0]
#
#     rdm = np.zeros((naos, naos, naos, naos))
#     for i in range(n):
#         for j in range(n):
#             rdm += get_two_rdm_contrib(ddets[i], dmos[i], cdets[j], cmos[j], sao)
#     return rdm

# def get_two_rdm_contrib(ddet, dmo, cdet, cmo, sao):
#     r"""
#     Get contributions to the 2-RDM for two non-orthogonal determinants (ddet and cdet), each with their MO coefficients
#     (dmo, cmo). 'd' labels the bra state and 'c' labels the ket state
#     :param ddet:
#     :param dmo:
#     :param cdet:
#     :param cmo:
#     :param sao:
#     :return:
#     """
#     naos = dmo.shape[0]
#
#     # Biorthogonalise orbitals
#     dtrf, ctrf, s = lowdin_pairing(dmo[:, [x for x, occno in enumerate(ddet[1:]) if occno == 1]],
#                                    sao,
#                                    cmo[:, [x for x, occno in enumerate(cdet[1:]) if occno == 1]])
#
#     # Find number of zeroes
#     m = len([x for x in s if x < THRESH])
#
#     # Find reduced overlap
#     redS = np.prod([x for x in s if x > THRESH])
#     M = get_unweighted_codensity(dtrf, dtrf, s) + get_unweighted_codensity(dtrf, ctrf, s) + \
#         get_weighted_codensity(dtrf, ctrf, s)
#     P = get_unweighted_codensity(dtrf, ctrf, s)
#
#     # Return zero matrix if m > 2
#     if m > 2:
#         return np.zeros((naos, naos, naos, naos))
#     elif m == 0:
#         return (np.einsum("rp,sq->pqrs", M, M, optimize='optimal') -
#                 np.einsum("sp,rq->pqrs", M, M, optimize='optimal')) * redS
#     elif m == 1:
#         return (np.einsum("rp,sq->pqrs", P, M, optimize='optimal') +
#                 np.einsum("rp,sq->pqrs", M, P, optimize='optimal') -
#                 np.einsum("sp,rq->pqrs", P, M, optimize='optimal') -
#                 np.einsum("sp,rq->pqrs", M, P, optimize='optimal')) * redS
#     elif m == 2:
#         return (np.einsum("rp,sq->pqrs", P, P, optimize='optimal') -
#                 np.einsum("sp,rq->pqrs", P, P, optimize='optimal')) * redS

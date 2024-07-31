r"""
Operators are assembled here.
Kets have the format [pf, a, b, ... n] where pf = Phase factor (1 or -1). a, b, ..., n are integers (0 or 1) for
unfilled and filled spin-orbitals, respectively.
"""
import numpy as np

THRESH = 1e-8   # Threshold for a value to be 0


def phase_factor(p, ket):
    total_filled_before = np.sum(ket[1:p])
    return (-1) ** total_filled_before


def create(p, ket):
    r"""
    Creation operator for the p-th spin orbital
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    else:
        if ket[p] == 1:
            return 0
        else:
            pf = phase_factor(p, ket)
            ket[p] = 1
            ket[0] *= pf
            return ket


def annihilate(p, ket):
    r"""
    Annihilation operator for the p-th spin orbital
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    else:
        if ket[p] == 0:
            return 0
        else:
            pf = phase_factor(p, ket)
            ket[p] = 0
            ket[0] *= pf
            return ket


def excitation(p, q, ket):
    r"""
    The one-electron excitation operator a^{\dagger}_{p} a_{q}
    """
    return create(p, annihilate(q, ket))


def overlap(bra, ket):
    r"""
    Find overlap between the bra and the ket
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    if type(bra) == int:
        if bra == 0:
            return 0
    if all(x == y for x, y in zip(bra[1:], ket[1:])):
        return 1 * bra[0] * ket[0]
    else:
        return 0


def get_generic_overlap(bras, kets, bra_coeffs, ket_coeffs):
    r"""
    Get generic overlap between two linear combinations
    """
    o = 0
    for i, bra_coeff in enumerate(bra_coeffs):
        if np.isclose(bra_coeff, 0, rtol=0, atol=THRESH):
            pass
        else:
            for j, ket_coeff in enumerate(ket_coeffs):
                if np.isclose(ket_coeff, 0, rtol=0, atol=THRESH):
                    pass
                else:
                    o += bra_coeff * ket_coeff * overlap(bras[i], kets[j])
    return o


def get_no_overlap(bra, ket, cross_overlap_matrix):
    r"""
    Calculates the non-orthogonal overlap of a bra and ket
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    if type(bra) == int:
        if bra == 0:
            return 0
    if sum(bra[1:]) != sum(ket[1:]):  # Different electron numbers
        return 0
    bra_idx = np.nonzero(bra[1:])[0]
    ket_idx = np.nonzero(ket[1:])[0]
    overlap_matrix = cross_overlap_matrix[bra_idx, :][:, ket_idx]
    return np.linalg.det(overlap_matrix) * bra[0] * ket[0]


def get_generic_no_overlap(bras, kets, bra_coeffs, ket_coeffs, cross_overlap_matrix):
    r"""
    Get generic non-orthogonal overlap between two linear combinations
    """
    o = 0
    for i, bra_coeff in enumerate(bra_coeffs):
        if np.isclose(bra_coeff, 0, rtol=0, atol=THRESH):
            pass
        else:
            for j, ket_coeff in enumerate(ket_coeffs):
                if np.isclose(ket_coeff, 0, rtol=0, atol=THRESH):
                    pass
                else:
                    o += bra_coeff * ket_coeff * get_no_overlap(bras[i], kets[j],
                                                                cross_overlap_matrix)
    return o


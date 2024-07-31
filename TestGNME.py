import numpy as np
import copy
from BasicOperators import create, annihilate, excitation, overlap
from GNME.GNMERDM import test_two_rdm_contrib, get_two_rdm_contrib

def get_two_rdm(bra, ket):
    r"""
    Obtain the 2-RDM using a brute-force approach
    \gamma^{pq}_{rs} = < A | a^{\dagger}_{p} a^{\dagger}_{q} a_{s} a_{r} | B >
    :param bra:
    :param ket:
    :return:
    """
    n_dim = len(ket) - 1  # Number of MOs
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    if type(bra) == int or type(ket) == int:  # No computation required if bra or ket is zero
        assert (bra == 0 or ket == 0)
        return two_rdm
    for p in range(n_dim):
        for q in range(n_dim):
            for r in range(n_dim):
                for s in range(n_dim):
                    mod_ket = create(p + 1,
                                     create(q + 1,
                                            annihilate(s + 1,
                                                       annihilate(r + 1,
                                                                  copy.deepcopy(ket)))))
                    if type(mod_ket) == int:
                        assert mod_ket == 0
                        two_rdm[p][q][r][s] = 0
                    else:
                        two_rdm[p][q][r][s] = overlap(bra, mod_ket)
    return two_rdm

a_old_det=[1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
a_new_det=[1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1]
#a_old_det = [1,1,0,1,0]
#a_new_det = [1,1,0,0,1]

#a_old_det = [1,1,0,1,0,1,0,1,0]
#a_new_det = [1,1,0,0,1,1,0,1,0]

identity = np.identity(len(a_old_det)-1)
tensor3=get_two_rdm_contrib(a_old_det,identity,a_new_det,identity,identity,identity)
tensor4=get_two_rdm(a_old_det,a_new_det)

tensor5=get_two_rdm_contrib(a_new_det,identity,a_old_det,identity,identity,identity)
tensor6=get_two_rdm(a_new_det,a_old_det)


print(np.allclose(tensor3,tensor6,rtol=0,atol=1e-6))
print(np.allclose(tensor4,tensor5,rtol=0,atol=1e-6))


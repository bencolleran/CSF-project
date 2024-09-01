import copy
import numpy as np
from BasicOperators import create, annihilate, excitation, overlap
import wfctn_2
from GNMERDM import get_one_rdm_contrib
from GNMERDM import get_two_rdm_contrib
import sys 
from contextlib import redirect_stdout
import opt_einsum as oe 

def single_electron_matrix():
    B=[3512,9679,4653,8472,6562]
    delta=[0,1536,4433,6167,6167]
    n=5
    zero_matrix = np.zeros((20,20))
    for i in range(4*n):
        zero_matrix[i][i]=delta[i%n]
    for i in range(n):
        zero_matrix[i+n][i]=B[i%5]
        zero_matrix[i][i+5]=B[i%5]
        zero_matrix[i+3*n][i+2*n]=B[i%5]
        zero_matrix[i+2*n][i+3*n]=B[i%5]
    return np.array(zero_matrix)
single_electron_matrix=single_electron_matrix()

def double_electron_tensor():
    tensor=np.zeros((20,20,20,20))
    n=5
    J=[2656,2743,2151,1756,395]
    #J=[1,1,1,1,1]
    #s1izs2iz
    for a in range(n):
        tensor[a][a+n][a][a+n]=J[a%n]*0.25
        tensor[a][a+3*n][a+3*n][a]=J[a%n]*0.25
        tensor[a+2*n][a+n][a+n][a+2*n]=J[a%n]*0.25
        tensor[a+2*n][a+3*n][a+2*n][a+3*n]=J[a%n]*0.25
    #s1i-s2i+
    for c in range(n):
        tensor[c+2*n][c+n][c+3*n][c]=J[c]*0.5
    #s1i+s2i-
    for d in range(n):
        tensor[d][d+3*n][d+n][d+2*n]=J[d]*0.5
    return tensor
double_electron_tensor=double_electron_tensor()

def get_one_rdm(bra, ket):
    r"""
    Obtain the 1-RDM corresponding to the bra and ket given
    < A | a^{\dagger}_{p} a_{q} | B >
    :param bra:
    :param ket:
    :return:
    """
    n_dim = len(ket) - 1  # Number of MOs
    one_rdm = np.zeros((n_dim, n_dim))
    for p in range(n_dim):
        for q in range(n_dim):
            # 1-ordering here
            mod_ket = excitation(p + 1, q + 1, copy.deepcopy(ket))
            if type(mod_ket) == int or type(bra) == int:  # The only int value it can take is zero
                assert (mod_ket == 0 or bra == 0)  # Just to be safe, we assert it
                one_rdm[p][q] = 0
            else:
                one_rdm[p][q] = overlap(bra, mod_ket)
    return one_rdm

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

def all_csf_couplings():
    CSF={'sextet':[0.5,1,1.5,2,2.5],'quintet':[0.5,1,1.5,2],'quartet1':[0.5,1,1.5,2,1.5],'quartet2':[0.5,1,1.5,1,1.5],'quartet3':[0.5,1,0.5,1,1.5],
    'quartet4':[0.5,0,0.5,1,1.5],'triplet1':[0.5,1,1.5,1],'triplet2':[0.5,1,0.5,1],'triplet3':[0.5,0,0.5,1],'doublet1':[0.5,1,0.5,1,0.5],'doublet2':[0.5,1,0.5,0,0.5],
    'doublet3':[0.5,0,0.5,1,0.5],'doublet4':[0.5,0,0.5,0,0.5],'doublet5':[0.5,1,1.5,1,0.5],'singlet1':[0.5,1,0.5,0],'singlet2':[0.5,0,0.5,0]}

    all_csfs=[key for key in CSF.keys()]
    all_csf_couplings=[['sextet','quintet']]
    for i in range(len(all_csfs[2:6])):
        for j in range(len(all_csfs[6:9])):
            all_csf_couplings+=[[all_csfs[i+2],all_csfs[j+6]]]
    for i in range(len(all_csfs[9:14])):
        for j in range(len(all_csfs[14:16])):
            all_csf_couplings+=[[all_csfs[i+9],all_csfs[j+14]]]
    return all_csf_couplings
all_csf_couplings=all_csf_couplings()

def index_coefficient(lst,lst_search):
    def deep_index(lst,lst_search):
        def deepest_index(lst,lst_search):
            for i in range(len(lst)):
                if all(x == y for x, y in zip(lst[i],lst_search)):
                    return i
        if deepest_index(lst[0],lst_search)!=None:
            return deepest_index(lst[0],lst_search)

    if deep_index(lst,lst_search)==None:
        return 0
    else:
        return lst[1][deep_index(lst,lst_search)]

dets=wfctn_2.determinants
hamiltonian_of_dets=np.zeros((len(dets),len(dets)))
task=[(i,j) for i in range(len(dets)) for j in range(i,len(dets))]
nmos = len(wfctn_2.lst2[0][0][0]) - 1
identity=np.identity(nmos)
off_diagonal_elements=[(i,j) for i in range(23) for j in range(i,23)]
hamiltonian_matrix=np.zeros((23,23))

def expectation_energy_of_dets(tple):
    i,j =tple
    E=oe.contract("pq,pq->",get_one_rdm_contrib(dets[j],identity,dets[i],identity,identity,identity), single_electron_matrix,backend='jax')
    +0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(dets[j],identity,dets[i],identity,identity,identity), double_electron_tensor,backend='jax')
    #E=0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(dets[j],identity,dets[i],identity,identity,identity), double_electron_tensor,backend='jax')
    return (i,j,E)


def generate_coefficients(k):
    coefficients=[index_coefficient(wfctn_2.lst2[k],dets[i]) for i in range(len(dets))]
    return coefficients

#form the determinant only hamiltonian
for k in range(len(task)):
    i, j, modified_value = expectation_energy_of_dets(task[k])
    hamiltonian_of_dets[i, j] = modified_value
    hamiltonian_of_dets[j, i] = modified_value

def matrix_element(tpl):
    a,b= tpl
    coefficient_bra=np.array(generate_coefficients(a))
    coefficient_ket=np.array(generate_coefficients(b))
    result=oe.contract('ij,j->i',hamiltonian_of_dets,coefficient_ket,backend='jax')
    e=oe.contract('i,i->',coefficient_bra,result,backend='jax')
    return (a,b,e)

def full_matrix():
    for k in range(len(off_diagonal_elements)):
        i, j, modified_value = matrix_element(off_diagonal_elements[k])
        hamiltonian_matrix[i, j] = modified_value
        hamiltonian_matrix[j, i] = modified_value
    print(hamiltonian_matrix)

with open('energies.txt', 'w') as f:
    # Redirect stdout to the file
    with redirect_stdout(f):
        # Call the function that prints output
        full_matrix()
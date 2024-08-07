import copy
import numpy as np
from BasicOperators import create, annihilate, excitation, overlap
#import wfctn_2
from GNMERDM import get_one_rdm_contrib
from GNMERDM import get_two_rdm_contrib
from Wavefunction_generator import single_site_csf
from Wavefunction_generator import double_site_csf
#from contextlib import redirect_stdout
import opt_einsum as oe 
'''
def single_electron_matrix():
    n=5
    matrix_1 = np.zeros((2*n,2*n))
    #for i in range(n):#-sz
    #    matrix_1[i][i]+=-0.5
    #    matrix_1[i+n][i+n]+=0.5
    tensor_contraction=np.zeros((2*n,2*n))
    for p in range(n):#delta_p_q,p=q
        tensor_contraction[p][p]+=0.25#p=q
        tensor_contraction[p][p]+=-0.25#p=q+n
        tensor_contraction[p+n][p+n]+=-0.25#p+n=q
        tensor_contraction[p+n][p+n]+=0.25#p+n=q+n
        #tensor_contraction[p][p]+=1
    return matrix_1+tensor_contraction
single_electron_matrix=single_electron_matrix()

def double_electron_tensor():
    n=5
    tensor=np.zeros(((2*n,2*n,2*n,2*n)))
    for p in range(n):
        for q in range(n):
            #tensor[p][q+n][p+n][q]+=-1#s+s-
            #tensor[p][q+n][q][p+n]+=1#
            tensor[p][q][p][q]+=-0.25#sz,always zero
            tensor[p][q][q][p]+=-0.25
            tensor[p][q+n][p][q+n]+=0.25
            tensor[p][q+n][q+n][p]+=0.25#multiple combinations?
            tensor[p+n][q][p+n][q]+=0.25
            tensor[p+n][q][q][p+n]+=0.25#
            tensor[p+n][q+n][p+n][q+n]+=-0.25#sz,always zero
            tensor[p+n][q+n][q+n][p+n]+=-0.25
    return tensor
double_electron_tensor=double_electron_tensor()
'''
#SzxSz
n=10
O=np.identity(n)
O*=0.25
#print(O)

A=np.identity(n)
for i in range(n):
    A[i][i]*=0.5
    if i>=n//2:
        A[i][i]*=-1
#print(A)
T=np.zeros((n,n,n,n))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                #T[p][r][q][s]=A[p][q]*A[r][s]+A[r][s]*A[p][q]#r->q,q->r
                #T[p][r][q][s]=A[p][s]*A[r][q]+A[p][s]*A[r][q]
                T[p][q][r][s]=A[p][r]*A[q][s]+A[p][r]*A[q][s]
#print(T)

#Sz
F=-1*A

#S+
C=np.zeros((n,n))
for p in range(n//2):
    C[p][p+n//2]=1
    #C[p+n//2][p]=-1
#print(C)

#S-
D=np.zeros((n,n))
for p in range(n//2):
    D[p+n//2][p]=1
    #D[p][p+n//2]=-1
#print(D)

O_s_plus_s_minus=np.identity(n)
for i in range(n//2):
    O_s_plus_s_minus[i+n//2][i+n//2]*=0
#print(O_s_plus_s_minus)
#O_s_plus_s_minus=np.zeros((n,n))

T_s_plus_s_minus=np.zeros((n,n,n,n))
for p in range(n):
    for r in range(n):
        for s in range(n):
            for q in range(n):
                #T_s_plus_s_minus[p][r][s][q]+=1*(C[p][q]*D[r][s]+C[r][s]*D[p][q])#correct
                T_s_plus_s_minus[p][r][q][s]+=0.25*(C[p][q]*D[r][s]+C[r][s]*D[p][q])
                T_s_plus_s_minus[r][p][q][s]+=-0.25*(C[p][q]*D[r][s]+C[r][s]*D[p][q])
                T_s_plus_s_minus[r][p][s][q]+=0.25*(C[p][q]*D[r][s]+C[r][s]*D[p][q])
                T_s_plus_s_minus[p][r][s][q]+=-0.25*(C[p][q]*D[r][s]+C[r][s]*D[p][q])


#single_electron_matrix=O+F#+O_s_plus_s_minus
#double_electron_tensor=T#+T_s_plus_s_minus
single_electron_matrix=O_s_plus_s_minus
double_electron_tensor=T_s_plus_s_minus
singlet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),-1/np.sqrt(2)]]
triplet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),1/np.sqrt(2)]]
#csf=triplet
#csf=singlet
csf=single_site_csf('sextet',1.5)
#print(csf)
#csf=double_site_csf('quintet',1)
#print(csf)
#csf=double_site_csf('quintet',1)
task=[(i,j) for i in range(len(csf[0])) for j in range(i,len(csf[0]))]
identity=np.identity((len(csf[0][0])-1))
hamiltonian_of_dets=np.zeros((len(csf[0]),len(csf[0])))

def expectation_energy_of_dets(tple):
    i,j =tple
    E=(oe.contract("pq,pq->",get_one_rdm_contrib(csf[0][j],identity,csf[0][i],identity,identity,identity), single_electron_matrix,backend='jax')
    +0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity), double_electron_tensor,backend='jax'))
    return (i,j,E)



#form the determinant only hamiltonian
for k in range(len(task)):
    i, j, modified_value = expectation_energy_of_dets(task[k])
    hamiltonian_of_dets[i][j] = modified_value
    hamiltonian_of_dets[j][i] = modified_value
#print(hamiltonian_of_dets)

coefficient_bra=np.array(csf[1])
coefficient_ket=np.array(csf[1])
result=oe.contract('ij,j->i',hamiltonian_of_dets,coefficient_ket,backend='jax')
e=oe.contract('i,i->',coefficient_bra,result,backend='jax')
print(e)



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

def get_mc_one_rdm(bras,kets,bra_coeffs,ket_coeffs):
    r"""
    Obtain the 1-RDM corresponding to a multi-configurational wavefunction
    < A | a^{\dagger}_{p} a_{q} | B >
    :param kets:
    :param coeffs:
    :return:
    """
    n_dim = len(kets[0]) - 1  # Number of MOs
    n_kets = len(kets)  # Number of states
    n_bras=len(bras)
    assert n_kets == len(ket_coeffs)  # Each ket must have a coefficient
    one_rdm = np.zeros((n_dim, n_dim))
    for i in range(n_bras):
        for j in range(n_kets):
            one_rdm += get_one_rdm(bras[i], kets[j]) * bra_coeffs[i] * ket_coeffs[j]
    return one_rdm

def get_mc_two_rdm(bras,kets,bra_coeffs,ket_coeffs):
    r"""
    Obtain the 2-RDM corresponding to a multi-configurational wavefunction
    < A | a^{\dagger}_{p} a^{\dagger}_{r} a_{s} a_{q} | B >
    :param kets:
    :param coeffs:
    :return:
    """
    n_dim = len(kets[0]) - 1  # Number of MOs
    n_kets = len(kets)  # Number of states
    assert n_kets == len(ket_coeffs)  # Each ket must have a coefficient
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for i in range(n_kets):
        for j in range(n_kets):
            two_rdm += get_two_rdm(bras[i], kets[j]) * bra_coeffs[i] * ket_coeffs[j]
    return two_rdm

'''
O_s_minus_s_plus=np.identity(n)
for i in range(n//2):
    O_s_minus_s_plus[i][i]*=0

T_s_minus_s_plus=np.zeros((n,n,n,n))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                #T_s_minus_s_plus[p][q][r][s]+=0.5*(D[p][s]*C[q][r]+D[q][r]*C[p][s])#r->q,q->s
                T_s_minus_s_plus[p][r][q][s]+=D[p][s]*C[r][q]+D[r][q]*C[p][s]#best
                #T_s_minus_s_plus[p][q][r][s]+=-0.5*(D[p][r]*C[q][s]+D[q][s]*C[p][r])#best
                #T_s_minus_s_plus[r][p][s][q]+=1*D[r][p]*C[s][q]+1*D[s][q]*C[r][p]

single_electron_matrix=#O-F+O_s_minus_s_plus
double_electron_tensor=T+T_s_minus_s_plus
#single_electron_matrix=0.5*(O-F+O_s_minus_s_plus)+0.5*(O+F+O_s_plus_s_minus)
#double_electron_tensor=0.5*(T+T_s_minus_s_plus)+0.5*(T+T_s_plus_s_minus)

'''
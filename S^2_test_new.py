import copy
import numpy as np
from BasicOperators import create, annihilate, excitation, overlap
#import wfctn_2
from GNMERDM import get_one_rdm_contrib
from GNMERDM import get_two_rdm_contrib
from GNMERDM import get_gnme_rdm
from Wavefunction_generator_old import single_site_csf
from Wavefunction_generator_old import double_site_csf
#from contextlib import redirect_stdout
import opt_einsum as oe 
from Wavefunction_generator_parametrised import csf_3


singlet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),-1/np.sqrt(2)]]
triplet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),1/np.sqrt(2)]]
csf=csf_3('doublet2',0.5)
#csf=triplet
#csf=singlet
#csf=single_site_csf('sextet',0.5)
#rint(csf)
#csf=double_site_csf('quintet',2)
print(csf)
print(csf_3('doublet1',0.5))
print(csf_3('doublet2',0.5))
#csf=double_site_csf('quintet',1)
task=[(i,j) for i in range(len(csf[0])) for j in range(i,len(csf[0]))]
identity=np.identity((len(csf[0][0])-1))
hamiltonian_of_dets=np.zeros((len(csf[0]),len(csf[0])))





def expectation_energy_of_dets(tple):
    nmos=6
    i,j =tple
    Sum1=0
    for p in range(nmos//2):
        Sum1+=get_one_rdm_contrib(csf[0][j],identity,csf[0][i],identity,identity,identity)[p][p]
    Sum2=0
    for p in range(nmos//2):
        for q in range(nmos//2):
            print(get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity)[p][q+nmos//2][p+nmos//2][q],p,q)
            Sum2+=get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity)[p][q+nmos//2][p+nmos//2][q]
    E=Sum1-Sum2
    return (i,j,E)
#print(get_rdm_contrib(csf[0][2],identity,csf[0][0],identity,identity,identity))

#form the determinant only hamiltonian
for k in range(len(task)):
    i, j, modified_value = expectation_energy_of_dets(task[k])
    #print(i,j,modified_value)
    hamiltonian_of_dets[i][j] = modified_value
    hamiltonian_of_dets[j][i] = modified_value
print(hamiltonian_of_dets)

coefficient_bra=np.array(csf[1])
coefficient_ket=np.array(csf[1])
#result=oe.contract('ij,j->i',hamiltonian_of_dets,coefficient_ket,backend='jax')
e=oe.contract('i,ij,j->',coefficient_bra,hamiltonian_of_dets,coefficient_ket,backend='jax')
print(e)
#print(coefficient_bra)
#print(coefficient_ket)
#print(np.einsum("i,ij,j->",coefficient_bra,hamiltonian_of_dets,coefficient_ket))

bras=csf_3('doublet1',0.5)[0]+csf_3('doublet2',0.5)[0]+csf_3('quartet',0.5)[0]+csf_3('quartet',1.5)[0]
bras_coeffs=csf_3('doublet1',0.5)[1]+csf_3('doublet2',0.5)[1]+csf_3('quartet',0.5)[1]+csf_3('quartet',1.5)[1]
identity=np.identity(6)
mos=[identity]
#print('trace=')

#print(np.einsum("ipip",get_gnme_rdm(bras,bras,bras_coeffs,bras_coeffs,mos,identity,identity,2)))
'''
T=np.zeros((6,6))
for p in range(6):
    for q in range(6):
        T[p][q]=get_gnme_rdm(bras,bras,bras_coeffs,bras_coeffs,mos,identity,identity,2)[p][q][p][q]
print(T)
'''

#T=np.zeros((6,6))
#for p in range(6):
#    for q in range(6):
#        T[p][q]=get_gnme_rdm(csf[0],csf[0],csf[1],csf[1],mos,identity,identity,2)[p][q][q][p]
#print(T)

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


#T=np.zeros((6,6))
#for p in range(6):
#    for q in range(6):
#        T[p][q]=get_mc_two_rdm(csf[0],csf[0],csf[1],csf[1])[p][q][q][p]
#print(T)

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


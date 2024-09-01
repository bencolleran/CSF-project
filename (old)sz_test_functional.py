import copy
import numpy as np
from BasicOperators import create, annihilate, excitation, overlap
import wfctn_2
from GNMERDM import get_one_rdm_contrib
from GNMERDM import get_two_rdm_contrib
from Wavefunction_generator import single_site_csf
from Wavefunction_generator import double_site_csf
import sys 
from contextlib import redirect_stdout
import opt_einsum as oe 

def single_electron_matrix():
    n=2
    matrix_1 = np.zeros((2*n,2*n))
    for i in range(n):#-sz
        matrix_1[i][i]+=-0.5
        matrix_1[i+n][i+n]+=0.5
    tensor_contraction=np.zeros((2*n,2*n))
    for p in range(n):#delta_p_q,p=q
        tensor_contraction[p][p]+=0.25#p=q
        tensor_contraction[p][p]+=-0.25#p=q+n
        tensor_contraction[p+n][p+n]+=-0.25#p+n=q
        tensor_contraction[p+n][p+n]+=0.25#p+n=q+n
        tensor_contraction[p][p]+=1
    return matrix_1+tensor_contraction
single_electron_matrix=single_electron_matrix()

def double_electron_tensor():
    n=2
    tensor=np.zeros(((2*n,2*n,2*n,2*n)))
    for p in range(n):
        for q in range(n):
            tensor[p][q+n][p+n][q]+=-1#s+s-
            tensor[p][q+n][q][p+n]+=1#
            tensor[p][q][p][q]+=-0.25#sz,always zero
            tensor[p][q][q][p]+=-0.25
            tensor[p][q+n][p][q+n]+=0.25
            tensor[p][q+n][q+n][p]+=0.25#multiple combinations see the bible
            tensor[p+n][q][p+n][q]+=0.25
            tensor[p+n][q][q][p+n]+=0.25#
            tensor[p+n][q+n][p+n][q+n]+=-0.25#sz,always zero
            tensor[p+n][q+n][q+n][p+n]+=-0.25
    return tensor
double_electron_tensor=double_electron_tensor()

singlet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),-1/np.sqrt(2)]]
triplet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),1/np.sqrt(2)]]
csf=triplet
#csf=singlet
#csf=single_site_csf('sextet',2.5)
#csf=double_site_csf('quintet',1)
#print(csf[1])
task=[(i,j) for i in range(len(csf[1])) for j in range(i,len(csf[1]))]
identity=np.identity((2*len(csf[1])))
hamiltonian_of_dets=np.zeros((len(csf[1]),len(csf[1])))

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


coefficient_bra=np.array(csf[1])
coefficient_ket=np.array(csf[1])
result=oe.contract('ij,j->i',hamiltonian_of_dets,coefficient_ket,backend='jax')
e=oe.contract('i,i->',coefficient_bra,result,backend='jax')
print(e)



#print(matrix_element((0,0)))
#i, j, modified_value = matrix_element((0,0))
#a=np.zeros((1,1))
#a[i][j]=modified_value
#print(a)
#E=oe.contract("pq,pq->",csf_1,single_electron_matrix,backend='jax')
#+0.5*oe.contract("pqrs,pqrs->",csf_2, double_electron_tensor,backend='jax')
#print(E)
#a=oe.contract("pq,pq->",csf_1,single_electron_matrix,backend='jax')
#b=0.5*oe.contract("pqrs,pqrs->",csf_2, double_electron_tensor,backend='jax')
#print(a+b)

def new_expectation_energy_of_dets(det1,det2):
    E=oe.contract("pq,pq->",get_one_rdm_contrib(det1,identity,det2,identity,identity,identity), old_single_electron_matrix,backend='jax')
    #+0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(dets[i],identity,dets[j],identity,identity,identity), double_electron_tensor,backend='jax')
    #return (i,j,E)
    return E
det1=[1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
det2=[1,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1]

#print(new_expectation_energy_of_dets(det1,det2))
i=0
bag=get_two_rdm_contrib(det1,identity,det1,identity,identity,identity)
#print(get_two_rdm_contrib(det1,identity,det1,identity,identity,identity)[i+0][i+5][i+0][i+5])
#print(get_two_rdm_contrib(det1,identity,det1,identity,identity,identity)[i][i+15][i+15][i])
#print(get_two_rdm_contrib(det1,identity,det1,identity,identity,identity)[i+10][i+5][i+10][i+5])
#print(oe.contract("pqrs,pqrs->",bag, double_electron_tensor,backend='jax'))

a_old_det=[1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
a_new_det=[1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1]
s1minus_s2plus=get_two_rdm_contrib(a_old_det,identity,a_new_det,identity,identity,identity)

b_old_det=[1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
b_new_det=[1,1,1,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,1]
s1plus_s2minus=get_two_rdm_contrib(b_new_det,identity,b_old_det,identity,identity,identity)
one_s1plus_s2minus=get_one_rdm_contrib(b_old_det,identity,b_old_det,identity,identity,identity)
c=1
n=5
#print(s1minus_s2plus[c+2*n][c+n][c+3*n][c])

d=2
#print(s1plus_s2minus[d][d+3*n][d+n][d+2*n])
print(one_s1plus_s2minus)

#print(get_mc_two_rdm(csf[0],csf[0],csf[1],csf[1]))
#print(get_two_rdm_contrib(csf[0][0],identity,csf[0][1],identity,identity,identity)[1][2][3][0])
#print(expectation_energy_of_dets((0,1)))
#print(double_electron_tensor[2][1][3][0])
#print(double_electron_tensor[1])
#print(single_electron_matrix)
#print(oe.contract("pqrs,pqrs->",get_two_rdm_contrib(csf[0][1],identity,csf[0][0],identity,identity,identity),double_electron_tensor,backend='jax'))
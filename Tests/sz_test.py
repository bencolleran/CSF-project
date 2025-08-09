import copy
import numpy as np
from BasicOperators import create, annihilate, excitation, overlap
import wfctn_2
from SummerProject.GNMERDM import get_one_rdm_contrib
from SummerProject.GNMERDM import get_two_rdm_contrib
from Wavefunction_generator import single_site_csf
from Wavefunction_generator import double_site_csf
import sys 
from contextlib import redirect_stdout
import opt_einsum as oe 

def single_electron_matrix():
    n=5
    matrix = np.zeros((10,10))
    for i in range(n):
        matrix[i][i]=-0.5
        matrix[i+n][i+n]=0.5
    return matrix
single_electron_matrix=single_electron_matrix()
'''
def double_electron_tensor():
    tensor=np.zeros(((20,20,20,20)))
    n=5
    for p in range(2*n):
        for q in range(2*n):
            tensor[p][q+2*n][p+2*n][q]=1#s+s-
            tensor[p][q][q][p]=0.25#sz
            tensor[p][q+2*n][p][q+2*n]=0.25
            tensor[p+2*n][q][p+2*n][q]=0.25
            tensor[p+2*n][q+2*n][q+2*n][p+2*n]=0.25
    return tensor
double_electron_tensor=double_electron_tensor()
'''
def double_electron_tensor():
    tensor=np.zeros(((10,10,10,10)))
    n=5
    for p in range(n):
        for q in range(n):
            tensor[p][q+n][q][p+n]=1#s+s-
            tensor[p][q][p][q]=0.25#sz
            tensor[p][q+n][p][q+n]=0.25
            tensor[p+n][q][p+n][q]=0.25
            tensor[p+n][q+n][p+n][q+n]=0.25
    return tensor
double_electron_tensor=double_electron_tensor()



'''
def double_electron_tensor():
    tensor=np.zeros(((20,20,20,20)))
    n=5
    for p in range(2*n):
        for q in range(2*n):
            tensor[p][q+2*n][q][p+2*n]=1#s+s-
            tensor[p][q][p][q]=-0.25#sz
            tensor[p][q+2*n][p][q+2*n]=0.25
            tensor[p+2*n][q][p+2*n][q]=0.25
            tensor[p+2*n][q+2*n][p+2*n][q+2*n]=-0.25
    return tensor
double_electron_tensor=double_electron_tensor()
'''
#csf=wfctn_2.lst2[0]
#sao=np.identity(20)
#csf_1=get_gnme_rdm(csf[0],csf[0],csf[1],csf[1], [sao],sao,sao,1)
#csf_2=get_gnme_rdm(csf[0],csf[0],csf[1],csf[1], [sao],sao,sao,2)

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
'''
dets=wfctn_2.determinants
hamiltonian_of_dets=np.zeros((len(dets),len(dets)))
task=[(i,j) for i in range(len(dets)) for j in range(i,len(dets))]
nmos = len(wfctn_2.lst2[0][0][0]) - 1
identity=np.identity(nmos)
off_diagonal_elements=[(i,j) for i in range(23) for j in range(i,23)]
hamiltonian_matrix=np.zeros((23,23))
'''

csf=single_site_csf('sextet',2.5)
#csf=double_site_csf('quintet',1)
#print(csf[1])
task=[(i,j) for i in range(len(csf[0])) for j in range(i,len(csf[0]))]
identity=np.identity(10)
hamiltonian_of_dets=np.zeros((len(csf[0]),len(csf[0])))

def expectation_energy_of_dets(tple):
    i,j =tple
    E=oe.contract("pq,pq->",get_one_rdm_contrib(csf[0][j],identity,csf[0][i],identity,identity,identity), single_electron_matrix,backend='jax')
    +0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity), double_electron_tensor,backend='jax')
    #E=0.5*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity), double_electron_tensor,backend='jax')
    return (i,j,E)

#form the determinant only hamiltonian
for k in range(len(task)):
    i, j, modified_value = expectation_energy_of_dets(task[k])
    hamiltonian_of_dets[i, j] = modified_value
    hamiltonian_of_dets[j, i] = modified_value

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


    
    
    

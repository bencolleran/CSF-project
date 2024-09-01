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
from Wavefunction_generator_parametrised import csf_3
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
n=6
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
                T[p][q][r][s]=0.5*(A[p][r]*A[q][s]+A[p][r]*A[q][s])
#print(T)

#Sz
F=-1*A
#print(F)
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
                T_s_plus_s_minus[p][r][s][q]+=1*(C[p][q]*D[r][s]+C[r][s]*D[p][q])
                #T_s_plus_s_minus[p][r][q][s]+=-1*(C[p][q]*D[r][s])#+C[r][s]*D[p][q])
                #T_s_plus_s_minus[r][p][s][q]+=1*(C[p][q]*D[r][s])
                #T_s_plus_s_minus[r][p][s][q]+=-1*(C[p][q]*D[r][s])
                #T_s_plus_s_minus[p][r][q][s]+=-1*(C[p][q]*D[r][s]+C[r][s]*D[p][q])#correct
                #T_s_plus_s_minus[p][r][q][s]+=-1*(C[p][q]*D[r][s])#+C[r][s]*D[p][q])
                #T_s_plus_s_minus[r][p][q][s]+=1*(D[p][q]*C[r][s])#+C[r][s]*D[p][q])
                #T_s_plus_s_minus[r][p][s][q]+=-1*(C[p][q]*D[r][s])#+C[r][s]*D[p][q])
                #T_s_plus_s_minus[p][r][s][q]+=1*(D[p][q]*C[r][s])#+C[r][s]*D[p][q])
#print(T_s_plus_s_minus[5][0][0][5])
#print(T_s_plus_s_minus[5][0][5][0])
#print(T_s_plus_s_minus[0][5][0][5])
#print(T_s_plus_s_minus[0][5][5][0])
#print(T_s_plus_s_minus[6][0][1][5])
#print(T_s_plus_s_minus[6][0][5][1])
#print(T_s_plus_s_minus[0][6][1][5])
#print(T_s_plus_s_minus[0][6][5][1])
#print(T_s_plus_s_minus[11][6][16][1])
#tensor3=T_s_plus_s_minus
#print(tensor3[11][6][16][1]-tensor3[6][11][16][1]-tensor3[11][6][1][16]+tensor3[6][11][1][16])
#two_e=0.5*(np.einsum("pq,rs->prqs",C,D)-np.einsum("pq,rs->prsq",C,D)+np.einsum("pq,rs->rpqs",C,D)-np.einsum("pq,rs->rpqs",C,D))
two_e=np.einsum("pq,rs->prqs",C,D)
#two_e=np.zeros((n,n,n,n))
for p in range(n):
    for r in range(n):
        for s in range(n):
            for q in range(n):
                two_e[p][r][q][s]*=-1**(p+q)
two_e=np.einsum("pq,rs->prsq",C,D)
#two_e=np.einsum("pq,rs->prsq",C,D)

G=np.identity(n)
for i in range(n//2):
    G[i][i]*=0
#print(G)
print("here")
print(C[2][5])
print(C)
print(D[3][0])
print(D)

testG = np.einsum("qr,pq,rs->ps",G,C,D)
#print("HERE")
#print(testG)

one_e=np.zeros((n,n))
for p in range(n):
    for r in range(n):
        for s in range(n):
            for q in range(n):
                one_e[p][s]=G[q][r]*C[p][q]*D[r][s]
#two_e[0][5][2][3]=-1
#print(one_e)
print(two_e[5][0][2][3])
print(two_e[5][0][3][2])
print(two_e[0][5][2][3])
print(two_e[0][5][3][2])
#print(T_s_plus_s_minus[6][0][1][5])
#print(T_s_plus_s_minus[6][0][5][1])
#print(T_s_plus_s_minus[0][6][1][5])
#print(T_s_plus_s_minus[0][6][5][1])


#single_electron_matrix=np.zeros((n,n))+O+F+O_s_plus_s_minus
#double_electron_tensor=T_s_plus_s_minus+T
#print(single_electron_matrix)
#print(double_electron_tensor)
#single_electron_matrix=O_s_plus_s_minus
#double_electron_tensor=T_s_plus_s_minus
single_electron_matrix=+np.zeros((n,n))+testG+O+F#+one_e
double_electron_tensor=np.zeros((n,n,n,n))+two_e+T
singlet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),-1/np.sqrt(2)]]
triplet=[[[1,1,0,0,1],[1,0,1,1,0]],[1/np.sqrt(2),1/np.sqrt(2)]]
csf=csf_3('doublet2',0.5)
#csf=triplet
#csf=singlet
#csf=single_site_csf('sextet',0.5)
#rint(csf)
#csf=double_site_csf('quintet',2)
print(csf)
#csf=double_site_csf('quintet',1)
task=[(i,j) for i in range(len(csf[0])) for j in range(i,len(csf[0]))]
identity=np.identity((len(csf[0][0])-1))
hamiltonian_of_dets=np.zeros((len(csf[0]),len(csf[0])))

def expectation_energy_of_dets(tple):
    i,j =tple
    E=(oe.contract("pq,pq->",get_one_rdm_contrib(csf[0][j],identity,csf[0][i],identity,identity,identity), single_electron_matrix,backend='jax')
    +1*oe.contract("pqrs,pqrs->",get_two_rdm_contrib(csf[0][j],identity, csf[0][i],identity,identity,identity), double_electron_tensor,backend='jax'))
    return (i,j,E)
    #print(oe.contract("pq,pq->",get_one_rdm_contrib(csf[0][j],identity,csf[0][i],identity,identity,identity)))

#form the determinant only hamiltonian
for k in range(len(task)):
    i, j, modified_value = expectation_energy_of_dets(task[k])
    #print(i,j,modified_value)
    hamiltonian_of_dets[i][j] = modified_value
    hamiltonian_of_dets[j][i] = modified_value
print(hamiltonian_of_dets)

coefficient_bra=np.array(csf[1])
coefficient_ket=np.array(csf[1])
result=oe.contract('ij,j->i',hamiltonian_of_dets,coefficient_ket,backend='jax')
e=oe.contract('i,i->',coefficient_bra,result,backend='jax')
print(e)
print(np.einsum("i,ij,j->",coefficient_bra,hamiltonian_of_dets,coefficient_ket))




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

print(get_two_rdm_contrib(csf[0][2],identity,csf[0][0],identity,identity,identity)[5][0][2][3])
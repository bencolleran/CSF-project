r"""this is a script which uses the previously generated wfctns to generate all RDMs and from them the overall hamiltonian
matrix. This is incredibly slow and inefficient and Giant_hamiltonian.py and New_hamiltonian.py are huge improvements.
"""

import sys
import copy
import math
import profile
import itertools
import numpy as np
from sympy import S
from sympy.physics.wigner import clebsch_gordan
from BasicOperators import create, annihilate, excitation, overlap
import wfctn_1
import wfctn_2

#parameters
J=[2656,2743,2151, 1756,395]
B=[3512,9679,4653,8472,6562]
delta=[0,1536,4433,6167,6167]

def single_electron_matrix():
    B=[3512,9679,4653,8472,6562]
    delta=[0,1536,4433,6167,6167]
    zero_matrix = np.zeros((20,20))
    for i in range(20):
        zero_matrix[i][i]=delta[i%5]
    for i in range(20):
        zero_matrix[i-5][i]=B[i%5]
        zero_matrix[i][i-5]=B[i%5]
    return np.array(zero_matrix)
single_electron_matrix=single_electron_matrix()

def double_electron_tensor():
    tensor=np.zeros((20,20,20,20))
    J=[2656,2743,2151, 1756,395]
    m=[1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1]
    for a in range(15):#s1izs2iz
        tensor[a+5][a][a+5][a]=J[a%5]*0.25*m[a]
        tensor[a+5][a][a][a+5]=J[a%5]*0.25*m[a]
        tensor[a][a+5][a][a+5]=J[a%5]*0.25*m[a]
        tensor[a][a+5][a+5][a]=J[a%5]*0.25*m[a]
    for b in range(5):#s1izs2iz
        tensor[b+15][b][b+15][b]=J[b]*-0.25
        tensor[b+15][b][b][b+15]=J[b]*-0.25
        tensor[b][b+15][b][b+15]=J[b]*-0.25
        tensor[b][b+15][b+15][b]=J[b]*-0.25
    for c in range(5):#s1i-s2i+
        tensor[c+10][c+5][c][c+15]=J[c]*0.5
        tensor[c+10][c+5][c+15][c]=J[c]*0.5
        tensor[c+5][c+10][c][c+15]=J[c]*0.5
        tensor[c+5][c+10][c+15][c]=J[c]*0.5
    for d in range(5):#s1i+s2i-
        tensor[d][d+15][d+10][d+5]=J[d]*0.5
        tensor[d][d+15][d+5][d+10]=J[d]*0.5
        tensor[d+15][d][d+10][d+5]=J[d]*0.5
        tensor[d+15][d][d+5][d+10]=J[d]*0.5
    return tensor
double_electron_tensor=double_electron_tensor() 

from GNMERDM import get_gnme_rdm
import wfctn_1
import wfctn_2
cs=wfctn_1.lst1[0]

def rdm1(i,j):
    bra=wfctn_1.lst1[i]
    ket=wfctn_1.lst1[j]
    naos = len(cs[0][0]) - 1
    rdm1= get_gnme_rdm(bra[0], ket[0], bra[1], ket[1], [np.identity(naos)], np.identity(naos), np.identity(naos), 1)
    return rdm1

def rdm2(i,j):
    bra=wfctn_2.lst2[i]
    ket=wfctn_2.lst2[j]
    naos = len(cs[0][0]) - 1
    rdm2= get_gnme_rdm(bra[0], ket[0], bra[1], ket[1], [np.identity(naos)], np.identity(naos), np.identity(naos), 2)
    return rdm2
    
def matrix_element(i,j):    
    E=np.einsum("pq,pq->",rdm1(i,j), single_electron_matrix,optimize='optimal')
    +0.5*np.einsum("pqrs,pqrs->",rdm2(i,j), double_electron_tensor,optimize='optimal')
    return E

def process_element(task):
    i,j =task
    modified_element=matrix_element(i,j)
    return (i,j,modified_element)

hamiltonian_matrix=np.zeros((2,2))
test_task=[(i,j) for i in range(2) for j in range(i,2)]

profile.run('rdm1(0,0)')

for k in range(3):
    i, j, modified_value = process_element(test_task[k])
    hamiltonian_matrix[i, j] = modified_value
    hamiltonian_matrix[j, i] = modified_value
    #print(hamiltonian_matrix.tolist())
#print(hamiltonian_matrix.tolist())




#hamiltonian_matrix=np.zeros((23,23))
#tasks=[(i,j) for i in range(23) for j in range(i,23)]
'''
import multiprocessing
if __name__ == '__main__':
    num_processes=multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_element, test_task)
    print(results)
    for k in range(len(results)):
        i, j, modified_value = results[k]
        hamiltonian_matrix[i, j] = modified_value
        hamiltonian_matrix[j, i] = modified_value
        print(hamiltonian_matrix.tolist())

print(hamiltonian_matrix.tolist())
'''
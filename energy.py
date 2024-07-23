import sys
import copy
import math
import itertools
import numpy as np
from sympy import S
from sympy.physics.wigner import clebsch_gordan
from BasicOperators import create, annihilate, excitation, overlap

def bubbleSort(arr):
    r"""
    Basic structure of the code taken from https://www.geeksforgeeks.org/python-program-for-bubble-sort/
    :param arr:
    :return:
    """
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    perm = 0  # Number of permutations
    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                perm += 1

        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return perm
    return perm

def get_phase_factor(alpha_idxs, beta_idxs):
    r"""
    From the orbital representation of a determinant [alpha_idxs], [beta_idxs], find the corresponding
    phase factor of said determinant.
    """
    double_occ = list(set(alpha_idxs).intersection(set(beta_idxs)))
    alpha_singly_occ = list(set(alpha_idxs) - set(double_occ))
    beta_singly_occ = list(set(beta_idxs) - set(double_occ))
    cur_order = alpha_singly_occ + beta_singly_occ
    perm = bubbleSort(cur_order)
    if perm % 2 == 0:   # Even permutation
        return 1
    else:
        return -1

def get_cg(j1, j2, j, m1, m2, m, analytic=False):
    r"""
    Get Clebsch-Gordon coefficients. Calculated using Sympy.
    :param j1: Spin of state 1
    :param j2: Spin of state 2
    :param j:  Spin of coupled state
    :param m1: Spin projection of state 1
    :param m2: Spin projection of state 2
    :param m:  Spin projection of coupled state
    :param analytic: :bool: if True, return analytic expression for the Clebsch-Gordon coefficient
    :return:   :float: Clebsch-Gordon coefficient
    """
    expr = clebsch_gordan(S(int(2 * j1)) / 2, S(int(2 * j2)) / 2, S(int(2 * j)) / 2, S(int(2 * m1)) / 2,
                          S(int(2 * m2)) / 2, S(int(2 * m)) / 2)
    if analytic:
        return expr
    else:
        return expr.evalf()

def get_general_tensorprod(j1, j2, j, m):
    r"""
    For a target COUPLED spin state of spin quantum number j with spin projection m,
    compute the necessary linear combination needed from states of spins j1 and j2
    :param j1: Spin of state 1
    :param j2: Spin of state 2
    :param j:  Spin of coupled state
    :param m:  Spin projection of coupled state
    :return:   List of List[float, float, float, float] in [j1, m1, j2, m2] of the states required for coupling
    """
    # We shall work in half-integer steps
    j1 = int(2 * j1)
    j2 = int(2 * j2)
    j = int(2 * j)
    m = int(2 * m)
    assert abs(j1 - j2) <= j <= j1 + j2, "Impossible set of spin quantum numbers"
    states_required = []
    for m1 in range(-j1, j1 + 1, 2):  # m goes in integer steps
        for m2 in range(-j2, j2 + 1, 2):
            if m1 + m2 == m:
                states_required.append([j1 / 2, m1 / 2, j2 / 2, m2 / 2])
    return states_required

def take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg):
    r"""
    Take the tensor product of the kets and cofficients on different sites. Multiply coefficient product by the
    Clebsch-Gordon coefficient.
    :param kets_a:      List of List[int]. List[int] has format: [pf, a, a, a, ..., b, b, ...]. pf = Phase factor,
                        a and b are alpha and beta occupations, respectively (0 for empty, 1 for filled)
    :param coeffs_a:    1D :np.ndarray: Coefficient of ket_a
    :param kets_b:      As kets_a
    :param coeffs_b:    As coeffs_a
    :param cg:          :float: Clebsch-Gordon coefficient
    :return:            List of List[int] of the coupled state
    """
    new_kets = []
    new_coeffs = []
    for a, ket_a in enumerate(kets_a):
        for b, ket_b in enumerate(kets_b):
            na = (len(ket_a)-1) // 2
            nb = (len(ket_b)-1) // 2
            pf = ket_a[0] * ket_b[0]
            new_ket = [pf] + ket_a[1:na+1] + ket_b[1:nb+1] + ket_a[na+1:] + ket_b[nb+1:]
            new_coeff = float(coeffs_a[a] * coeffs_b[b] * cg)
            new_kets.append(new_ket)
            new_coeffs.append(new_coeff)
    return [new_kets, new_coeffs]

def get_coupling_coefficient(Tn, Pn, tn, pn):
    r"""
    Computes the coupling coefficient C_{tn, pn}^{Tn, Pn}
    :param Tn:
    :param Pn:
    :param tn:
    :param pn:
    :return:
    """
    # This is a forbidden case
    if Tn < np.abs(Pn):
        return 0
    if np.isclose(0.5, tn, rtol=0, atol=1e-10):
        return np.sqrt((Tn + 2 * pn * Pn) / (2 * Tn))
    elif np.isclose(-0.5, tn, rtol=0, atol=1e-10):
        return -2 * pn * np.sqrt((Tn + 1 - 2 * pn * Pn) / (2 * (Tn + 1)))
    else:
        print("A coefficient requested is invalid. Exiting.")
        sys.exit(1)

def get_total_coupling_coefficient(det, csf):
    r"""
    Gets the overlap between the determinant and the CSF. This is the coefficient d of the determinant in the CSF.
    :param det:
    :param csf:
    :return:
    """
    total_coeff = 1
    assert len(det) == len(csf), "Number of orbitals in determinant and CSF are not the same. Check again."
    for i in range(1, len(det)):
        Tn = csf[i]
        Pn = det[i]
        tn = csf[i] - csf[i - 1]
        pn = det[i] - det[i - 1]
        total_coeff = total_coeff * get_coupling_coefficient(Tn, Pn, tn, pn)
    return total_coeff

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

def constraints_singly(perm,n):#filters all permutations with n singly occupied orbitals
    val=[1 for i in range(len(perm[:5])) if perm[i]!=perm[i+5]]
    if sum(val)>=n:
        return True
    else:
        return False

def perms(n,m):#generates all ON vectors with a given m
    nums={5:[1,1,1,1,1,0,0,0,0,0],6:[1,1,1,1,1,1,0,0,0,0]}
    all_perms = itertools.permutations(nums[n])
    unique_perms = set(all_perms)
    lst1 = [list(perm) for perm in unique_perms]
    lst2=[lst1[j] for j in range(len(lst1)) if constraints_singly(lst1[j],(10-sum(nums[n])))]
    lst3=[lst2[k] for k in range(len(lst2)) if 0.5*(sum(lst2[k][:5])-sum(lst2[k][5:]))==m]
    on_vectors=[[get_phase_factor(lst3[l][:5],lst3[l][5:])]+lst3[l] for l in range(len(lst3))]
    return on_vectors#with phase factor in [0]

def genealog(lst):#convert a on vector to a genealogical spin coupling
    n=0
    lst1=[]
    for i in range(len(lst[:5])):
        if lst[i+1]!=lst[i+6]:
            n+=0.5*(lst[i+1]-lst[i+6])
            lst1.append(n)
    return lst1

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

CSF={'sextet':[0.5,1,1.5,2,2.5],'quintet':[0.5,1,1.5,2],'quartet1':[0.5,1,1.5,2,1.5],'quartet2':[0.5,1,1.5,1,1.5],'quartet3':[0.5,1,0.5,1,1.5],
    'quartet4':[0.5,0,0.5,1,1.5],'triplet1':[0.5,1,1.5,1],'triplet2':[0.5,1,0.5,1],'triplet3':[0.5,0,0.5,1],'doublet1':[0.5,1,0.5,1,0.5],'doublet2':[0.5,1,0.5,0,0.5],
    'doublet3':[0.5,0,0.5,1,0.5],'doublet4':[0.5,0,0.5,0,0.5],'doublet5':[0.5,1,1.5,1,0.5],'singlet1':[0.5,1,0.5,0],'singlet2':[0.5,0,0.5,0]}

def coupled_states(c,n):#n is the number of sites with the 6 electrons, n=2 for linear combination
    states=all_csf_couplings()
    j=0.5
    m=0.5
    j1=CSF[states[c][0]][-1]
    j2=CSF[states[c][1]][-1]
    g=get_general_tensorprod(j1, j2, j, m)
    s=[[],[]]
    if n==1:
        for x in range(len(g)):
            coeffs_a=[get_total_coupling_coefficient(genealog(perms(5,g[x][1])[a]), CSF[states[c][0]]) for a in range(len(perms(5,g[x][1])))]
            coeffs_b=[1/math.sqrt(5)*get_total_coupling_coefficient(genealog(perms(6,g[x][3])[b]), CSF[states[c][1]]) for b in range(len(perms(6,g[x][3])))]
            kets_a=perms(5,g[x][1])
            kets_b=perms(6,g[x][3])
            cg=get_cg(j1, j2, j, g[x][1], g[x][3], m, analytic=False)
            t=take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg)
            s[0].extend(t[0])
            s[1].extend(t[1])
        return s
    if n==2:
        for x in range(len(g)):
            coeffs_a=[1/math.sqrt(math.sqrt(2))*get_total_coupling_coefficient(genealog(perms(5,g[x][1])[a]), CSF[states[c][0]]) for a in range(len(perms(5,g[x][1])))]
            coeffs_b=[1/math.sqrt(math.sqrt(2))*1/math.sqrt(5)*get_total_coupling_coefficient(genealog(perms(6,g[x][3])[b]), CSF[states[c][1]]) for b in range(len(perms(6,g[x][3])))]
            kets_a=perms(5,g[x][1])
            kets_b=perms(6,g[x][3])
            cg1=get_cg(j1, j2, j, g[x][1], g[x][3], m, analytic=False)
            cg2=get_cg(j2, j1, j, g[x][3], g[x][1], m, analytic=False)
            t=take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg1)
            u=take_csf_tensorprod(kets_b, coeffs_b, kets_a, coeffs_a, cg2)
            s[0].extend(t[0])
            s[0].extend(u[0])
            s[1].extend(t[1])
            s[1].extend(u[1])
        return s

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

def matrix_element(i,j):    
    E=(sum(sum(np.dot(get_mc_one_rdm(coupled_states(i,1)[0],coupled_states(j,1)[0],coupled_states(i,1)[1],coupled_states(j,1)[1]),single_electron_matrix)))
    +0.5*sum(sum(sum(sum(sum(np.dot(get_mc_two_rdm(coupled_states(i,2)[0],coupled_states(j,2)[0],coupled_states(i,2)[1],coupled_states(j,2)[1]), double_electron_tensor)))))))
    return E

def process_element(task):
    i,j,value=task
    modified_element=matrix_element(i,j)
    return (i,j,modified_element)

hamiltonian_matrix=np.zeros((23,23))
tasks=[(i,j,hamiltonian_matrix[i][j]) for i in range(23) for j in range(i,23)]

import multiprocessing
num_processes=multiprocessing.cpu_count()
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.map(process_element, tasks)

    for result in results:
        i, j, modified_value = result
        hamiltonian_matrix[i, j] = modified_value
        hamiltonian_matrix[j, i] = modified_value
        print(hamiltonian_matrix.tolist())

print(hamiltonian_matrix.tolist())
    
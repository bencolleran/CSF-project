import sys
import copy
import math
import itertools
import numpy as np
from sympy import S
from sympy.physics.wigner import clebsch_gordan

#calculate the number of permutations needed to shuffle the electrons in a SD, needed for get_phase_factor 
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

#returns a list of all states to produce an overall coupled state of j,m    
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

#returns the site coupled SDs and coefficients for two coupled sites
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

#CSF tools, gives coefficient of a single MO from a SD for a given csf in the genealogical coupling scheme e.g. [0,0.5,1,0.5] for a 4 MO basis
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

#CSF tools, gives coefficient of a single SD for a given csf in the genealogical coupling scheme e.g. [0,0.5,1,0.5] for a 4 MO basis
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

#constraint to satisfy all orbitals being either singly occupied or only one orbital being doubly occupied, dependent on n
def constraints_singly(perm,n,nmos):
    val=[1 for i in range(nmos//2) if perm[i]!=perm[i+nmos//2]]
    if sum(val)>=n:
        return True
    else:
        return False

#since alpha_indxs and beta_indxs are in a spatial orbital form, rather than an ON vector form, we must convert between them
def ONvector_to_spatial(lst,spin):
    #feed in an ON vector without the phase factor
    n=len(lst)//2
    spatial_orbs=[i for i in range(n)]
    if spin=='a':
        new_lst=[i for i in range(n) if lst[0:n][i]==1]
        return new_lst
    if spin=='b':
        new_lst=[i for i in range(n) if lst[n:2*n][i]==1]
        return new_lst

#generates all SDs with a given m value
def perms(n,m,nmos):
    #parameter nmos is the number of spin orbitals
    #parameter nums is two example dets
    nums={5:[1,1,1,1,1,0,0,0,0,0],6:[1,1,1,1,1,1,0,0,0,0],3:[1,1,1,0,0,0]}
    all_perms = itertools.permutations(nums[n])
    unique_perms = set(all_perms)
    lst1 = [list(perm) for perm in unique_perms]
    lst2=[lst1[j] for j in range(len(lst1)) if constraints_singly(lst1[j],(nmos-sum(nums[n])),nmos)]
    lst3=[lst2[k] for k in range(len(lst2)) if 0.5*(sum(lst2[k][:(nmos//2)])-sum(lst2[k][(nmos//2):]))==m]
    on_vectors=[[get_phase_factor(ONvector_to_spatial(lst3[l][:(nmos//2)],'a'),ONvector_to_spatial(lst3[l][(nmos//2):],'b'))]+lst3[l] for l in range(len(lst3))]
    return on_vectors#with phase factor in [0]

#convert an on vector to a genealogical spin coupling
def genealog(lst,nmos):
    n=0
    lst1=[]
    for i in range(len(lst[:(nmos//2)])):
        if lst[i+1]!=lst[i+1+(nmos//2)]:
            n+=0.5*(lst[i+1]-lst[i+1+(nmos//2)])
            lst1.append(n)
    return lst1

#all possible CSFs for the Fe2-S2 system
CSF={'sextet':[0.5,1,1.5,2,2.5],'quintet':[0.5,1,1.5,2],'quartet1':[0.5,1,1.5,2,1.5],'quartet2':[0.5,1,1.5,1,1.5],'quartet3':[0.5,1,0.5,1,1.5],
    'quartet4':[0.5,0,0.5,1,1.5],'triplet1':[0.5,1,1.5,1],'triplet2':[0.5,1,0.5,1],'triplet3':[0.5,0,0.5,1],'doublet1':[0.5,1,0.5,1,0.5],'doublet2':[0.5,1,0.5,0,0.5],
    'doublet3':[0.5,0,0.5,1,0.5],'doublet4':[0.5,0,0.5,0,0.5],'doublet5':[0.5,1,1.5,1,0.5],'singlet1':[0.5,1,0.5,0],'singlet2':[0.5,0,0.5,0]}

CSF_3={'quartet':[0.5,1,1.5],'doublet1':[0.5,0,0.5],'doublet2':[0.5,1,0.5]}
#generates all couplings of states to produce an overall state of S=0.5,M=0.5
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

#generates a wavefunction for a given index of the coupled states; list of list of SDs and coefficients
#n is the number of sites with the 6 electrons, n=2 for linear combination
def coupled_states(c,n):
    states=all_csf_couplings()
    nmos=10
    j=0.5
    m=0.5
    j1=CSF[states[c][0]][-1]
    j2=CSF[states[c][1]][-1]
    g=get_general_tensorprod(j1, j2, j, m)
    s=[[],[]]
    if n==1:
        for x in range(len(g)):
            coeffs_a=[get_total_coupling_coefficient(genealog(perms(5,g[x][1],nmos)[a],nmos), CSF[states[c][0]]) for a in range(len(perms(5,g[x][1],nmos)))]
            coeffs_b=[1/math.sqrt(5)*get_total_coupling_coefficient(genealog(perms(6,g[x][3],nmos)[b],nmos), CSF[states[c][1]]) for b in range(len(perms(6,g[x][3],nmos)))]
            kets_a=perms(5,g[x][1],nmos)
            kets_b=perms(6,g[x][3],nmos)
            cg=get_cg(j1, j2, j, g[x][1], g[x][3], m, analytic=False)
            t=take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg)
            s[0].extend(t[0])
            s[1].extend(t[1])
        return s
    if n==2:
        for x in range(len(g)):
            coeffs_a=[1/math.sqrt(math.sqrt(2))*get_total_coupling_coefficient(genealog(perms(5,g[x][1],nmos)[a],nmos), CSF[states[c][0]]) for a in range(len(perms(5,g[x][1],nmos)))]
            coeffs_b=[1/math.sqrt(math.sqrt(2))*1/math.sqrt(5)*get_total_coupling_coefficient(genealog(perms(6,g[x][3],nmos)[b],nmos), CSF[states[c][1]]) for b in range(len(perms(6,g[x][3],nmos)))]
            kets_a=perms(5,g[x][1],nmos)
            kets_b=perms(6,g[x][3],nmos)
            cg1=get_cg(j1, j2, j, g[x][1], g[x][3], m, analytic=False)
            cg2=get_cg(j2, j1, j, g[x][3], g[x][1], m, analytic=False)
            t=take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg1)
            u=take_csf_tensorprod(kets_b, coeffs_b, kets_a, coeffs_a, cg2)
            s[0].extend(t[0])
            s[0].extend(u[0])
            s[1].extend(t[1])
            s[1].extend(u[1])
        return s

#print(coupled_states(0,1))

def single_site_csf(csf,m):
    nmos=10
    coeffs=[get_total_coupling_coefficient(genealog(perms(5,m,10)[a], CSF.get(csf),nmos) for a in range(len(perms(5,m,10))))]
    kets=perms(5,m,10)
    csf=[kets]+[coeffs]
    return csf

def double_site_csf(csf,m):
    nmos=10
    coeffs=[get_total_coupling_coefficient(genealog(perms(6,m,10)[a],nmos), CSF.get(csf)) for a in range(len(perms(6,m,10)))]
    kets=perms(6,m,10)
    csf=[kets]+[coeffs]
    return csf
#print(double_site_csf('quintet',1))

#2-electron-csfs
singlet=[[[1,0,0,1],[0,1,1,0]],[1/np.sqrt(2),-1/np.sqrt(2)]]
triplet=[[[1,0,0,1],[0,1,1,0]],[1/np.sqrt(2),1/np.sqrt(2)]]

def csf_3(csf,m):
     nmos=6
     coeffs=[get_total_coupling_coefficient(genealog(perms(3,m,6)[a],nmos), CSF_3.get(csf)) for a in range(len(perms(3,m,6)))]
     kets=perms(3,m,6)
     csf=[kets]+[coeffs]
     return csf

#print(csf_3('doublet2',0.5))
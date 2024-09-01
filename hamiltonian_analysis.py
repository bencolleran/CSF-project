import numpy as np
from numpy.linalg import eig

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
#give filepath
filepath=r"C:\Users\DELL\Desktop\output_hamiltonian.txt"
with open(filepath, 'r') as file:
    lines = file.readlines()
a_lines=[lines[i].replace('\n',',') for i in range(len(lines))]
b_lines=[a_lines[i].replace('  ',',') for i in range(len(lines))]
c_lines=[b_lines[i].replace('[','') for i in range(len(lines))]
d_lines=[c_lines[i].replace(']','') for i in range(len(lines))]
e_lines=[d_lines[i].replace(' ',',') for i in range(len(lines))]
list_of_lists = [s.split(',') for s in e_lines]
total_list=[]
for j in range(len(list_of_lists)):
    f_list=[float(list_of_lists[j][i]) for i in range(len(list_of_lists[j])) if list_of_lists[j][i]!='']
    total_list+=f_list
#print(total_list)
num_rows = 23
num_cols = 23
# Split the flat list into a list of lists
matrix = [total_list[i*num_cols:(i+1)*num_cols] for i in range(num_rows)]
#print(np.array(matrix))
a=np.array(matrix)


Evalue,Evector=eig(a)
gs=Evalue.tolist().index(min(Evalue))#eigenvalues are a list of ints eigenvectors are a list of lists of ints
print(gs)
#print(Evalue.tolist()[10])
#print(Evalue.tolist())
#print(lst)
corrected_energy=[Evalue.tolist()[i]-Evalue.tolist()[gs] for i in range(23)]
print(sorted(corrected_energy))
lst2=sorted(corrected_energy)[:10]
for k in range(10):
    m=corrected_energy.index(lst2[k])
    lst=[(Evector.tolist()[m][i]**2,all_csf_couplings[i]) for i in range(23) if Evector.tolist()[m][i]**2>0.1]
    #print(lst)
print(lst2)

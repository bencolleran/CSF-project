import numpy as np
vec=np.array([2,-1,-1]/np.sqrt(6))
mat=np.array([[2,1,-1],[1,2,1],[-1,1,2]])
print(np.einsum('i,ij,j',vec,mat,vec)/1)

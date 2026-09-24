import numpy as np 

def productoMatrices(A,B):
  if A.shape[1] != B.shape[0]:
    return None 
    
  else:
    res = []
    for i in range(A.shape[0]):
      fila = []
      for j in range(B.shape[1]):
        suma = 0
        for k in range(A.shape[1]):
          suma += A[i][k]*B[k][j] # sumatoria aik * bkj
        fila.append(suma)
      res.append(fila) 
    return np.array(res, dtype=np.float64) 
      

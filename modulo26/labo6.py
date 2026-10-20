#%% MÉTODO DE LA POTENCIA !! 
import numpy as np 

def productoMatVec(A, v):
  if A.shape[1] != len(v):
    return None 
  res = []
  for i in range(A.shape[0]):
    a = 0
    for j in range(A.shape[1]):
      a += A[i][j] * v[j]
    res.append(a)
  return np.array(res, dtype=np.float64) 

def f2A(A,v): # es mult lambda por V y sabemos q lamba es vtAv
        w = productoMatVec(A, productoMatVec(A, v))
        return w / norma(w, 2) 

def metpot2k(A, tol=1e-15,K=1000):
  """
  A: matriz nxn
  tol: tolerancia en la dif entre un paso y el sig de la estimacion del autovect
  K: num maximo de iteraciones a realizarse
  devuelve vector V, autovalor lamda y cant de iteraciones k
  """
  if not simetrica(A):
    return None 
  n = A.shape[0]
  v1 = np.random.rand(n) 
  v1 = v1 / norma(v1, 2)
  v2 = f2A(v1) 
  e = producto(v2, v1, True)
  k = 0 
  while abs(e-1) > tol and k < K:
    v1 = v2 
    v2 = f2A(v1) 
    e = producto(v2, v1, True) 
    k = k + 1
  vector = productoMatVec(A, v2)
  l = producto(v2, vector, True) 
  return v2, l, k

#%% DIAGONALIZACION CON MP Y RH !! 

def diagRH(A, tol=1e-15,K=1000): 
  if not simetrica(A):
    return None 
  v, l = metpot2k(A, tol, K)
  H = # hacer este 
  if n == 2:
    S = H 
    D = # hacer este 
  else: 
    B = # hacer 
    A = B[2:n, 2:n] 
    S, D2 = diagRH(A, tol, K) 
    D1 = # armarlo 
    S = # armarlo 
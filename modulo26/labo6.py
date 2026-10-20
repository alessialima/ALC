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
  v, l , e = metpot2k(A, tol, K)
  if abs(e) > tol: 
        return np.eye(n), np.diag(np.diag(A))
    
  e1 = np.zeros(n)
  e1[0] = 1
  u = e1 - v.flatten() 
  norm_u = norma2(u) 
    
  if norm_u < tol:  
        H = np.eye(n)
  else:
        u = u / norm_u # si me paso, normalizo 
        H = np.eye(n) - 2 * producto(u,u, False) # prod externo

  M = prod(H,prod(A,traspuesta(H))) # me olvide como se llamaba mi funcion q multiplicaba matriz x vector desp veo 
  B = M[1:, 1:]
  
  if n == 2:
    S = H 
    D = np.diag([l, B[0, 0]]) # se puede usar np diag?? creo q no ni idea
 else:
    B = M[1:n, 1:n]
    S1, D1 = diagRH(B, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = l
    D[1:n, 1:n] = D1

    S2 = np.eye(n)
    S2[1:n, 1:n] = S1
    S = H @ S2 # armar funcion q multiplica matrices 

  return S, D

#%% ver. tp

def diagRH(A,tol=1e-15,K=np.inf):
    #Función recursiva para diagonalizar
    #como usa HH y Metpot O(n³)


    n = A.shape[0]
    # caso base
    if (n == 1):
        return np.eye(1), A

    v_1d,autval,*_ = metpot2k(A,tol,K)
    v = v_1d.reshape(-1, 1)
    v_unitario = v * 1/norma(v,2)
    I = np.eye(n)
        
    D = np.zeros((n,n))
    e_1 = np.zeros((n,1))
    e_1[0] = 1.0
    u = e_1 - v_unitario
    norma_al_cuadrado = prodMatricial(traspuesta(u),u)

    # aca evita division por cero cuando e_1 es practicamente igual a v_unitario. Hv NO tiene que hacer nada entonces Hv = I
    if norma_al_cuadrado < tol:
        Hv = I 
    else:
        #construyo householder
        Hv = I - 2 * prodMatricial(u, traspuesta(u)) / norma_al_cuadrado
    #construyo matriz de autovectores
    B = prodMatricial(prodMatricial(Hv,A),traspuesta(Hv))
    # aca hice esto al cortar B para que el resultado sea matriz y no un numpy-array. 
    A2 = B[1:n, 1:n]
    S2, D2 = diagRH(A2,tol,K)
    # construyo la submatriz D con n-1 de dimension
    D[0][0] = autval
    D[1:,1:] = D2
    #construyo S de igual forma a D
    Q = identidad(n)
    Q[1:, 1:] = S2

    return prodMatricial(Hv, Q),D

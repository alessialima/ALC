# obs: lo q me la caga es mult de matrices !!!!! mas q nada




def svd_reducida(A, K=np.inf, tol=1e-15):
    m, n = A.shape
    if m < n:
        U, S, V = svd_reducida(traspuesta(A), K, tol)
        return V, S, U

    AtA = prodMatricial(traspuesta(A), A)
    V, D = diagRHsinCeros(AtA, tol, 1000) 
    autovalores = [D[i, i] for i in range(n)]

    rango = sum(1 for a in autovalores if a > tol * autovalores[0])
    if K != np.inf:
        rango = min(rango, int(K))
    if rango == 0:
        raise ValueError("La matriz no puede ser nula")

    V = V[:, 0:rango]
    S = np.array([autovalores[i] ** 0.5 for i in range(rango)])

    U = prodMatricial(A, V) / S   
    return U, S, V

#%% 

import numpy as np

#%% MÉTODO DE LA POTENCIA
def metpot(A, tol=1e-12, K=1000):
  n = A.shape[0]
  v = np.random.rand(n)
  v = v / norma(v, 2)
  k = 0
  while k < K:
    w = productoMatVec(A, v)
    nw = norma(w, 2)
    if nw < tol:                     # si A nula o con autoval nulo
      return v, 0.0, k
    w = w / nw
    e = abs(producto(w, v, True))
    v = w
    k = k + 1
    if abs(e - 1) < tol:
      break
  l = producto(v, productoMatVec(A, v), True)
  return v, l, k

#%% DIAGONALIZACIÓN CON MP Y RH 
def diagRH(A, r=None, tol=1e-12, K=1000, lam1=None):
  """
  Devuelve V (n x k) y lams (lista de k autovalores, de mayor a menor módulo),
  con k <= r. Corta antes si |lambda| < tol * |lambda_1|.
  """
  n = A.shape[0]
  if r is None or r > n:
    r = n

  v, l, _ = metpot(A, tol, K)
  if lam1 is None:
    lam1 = abs(l)
  if abs(l) <= tol * lam1 or lam1 == 0:        # si no hay mas autovalores *no nulos*
    return np.zeros((n, 0)), []

  if r == 1 or n == 1:
    V = np.zeros((n, 1))
    for i in range(n):
      V[i, 0] = v[i]
    return V, [l]
      
#HH
    
  u = -v
  u[0] = u[0] + 1
  nu = norma(u, 2)
  if nu < tol:                                  
    u = np.zeros(n)
  else:
    u = u / nu

  # B = (H A H)[1:,1:] = A - 2uw^t - 2wu^t + 4c uu^t, con w = Au, c = u^t w
  w = productoMatVec(A, u)
  c = producto(u, w, True)
  q = 2 * w - 2 * c * u
  B = A[1:, 1:] - producto(u[1:], q[1:], False) - producto(q[1:], u[1:], False)

  V1, L1 = diagRH(B, r - 1, tol, K, lam1)
  k = len(L1)

  # S = H * [[1,0],[0,V1]] = S2 - 2 u (S2^t u)^t
  S2 = np.zeros((n, 1 + k))
  S2[0, 0] = 1
  S2[1:, 1:] = V1
  V = S2 - 2 * producto(u, productoMatVec(traspuesta(S2), u), False)
  return V, [l] + L1

#%% SVD REDUCIDA
def svd_reducida(A, K=np.inf, tol=1e-12):
  m, n = A.shape
  if m < n:                                     
    U, S, V = svd_reducida(traspuesta(A), K, tol)
    return V, S, U

  AtA = prodMatricial(traspuesta(A), A)
  r = n if K == np.inf else min(int(K), n)
  V, lams = diagRH(AtA, r, tol, 1000)

  rango = len(lams)
  if rango == 0:
    raise ValueError("La matriz no puede ser nula")

  S = np.array([lams[i] ** 0.5 for i in range(rango)])
  U = prodMatricial(A, V) / S                   
  return U, S, V

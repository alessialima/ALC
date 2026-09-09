import numpy as np 

#%% calcula LU 

def calculaLU(A): 

    if A is None:
        return None, None, 0

    
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    if m != n:
        return None, None, 0 

    nops = 0 # nro de operaciones 
    pivote = 0 
    U = A.copy() 
    L = np.eye(m) 

    # pivoteo:
    for i in range(m-1):
        pivote = U[i][i]
        if abs(pivote) < 1e-5: # si es muy petit no sirve 
            return None, None, 0 

        for j in range(i + 1, m):
            coef = U[j][i] / pivote 
            U[j][i] = 0 
            nops += 1  # divido
            L[j][i] = coef 

            for k in range(i+1, m):
                U[j][k] -= coef * U[i][k]
                nops += 2 # mult y resto 

    return np.array(L), np.array(U), nops


#%% triangular sup / inf 

def res_tri(L, b, inferior=True):
    L = np.array(L, dtype=np.float64)
    x = len(b)*[0]

    if inferior:
        for i in range(len(b)):
            suma = 0 
            for j in range(i):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
        return x 

    else: 
        for i in range(len(b) - 1, -1, -1): # voy de abajo hacia arriba para q tarde menos (si hacia traspuesta era un quilombo)
            suma = 0 
            for j in range(i+1, len(b)):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
        return x 

#%% inversa (A)

def det(A): # determinante
    A = np.array(A, dtype=np.float64)
    m, n = A.shape 

    if m != n:
        return 0 
    elif m == 0:
        return 1
    elif m == 1:
        return A[0][0]
    elif m == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]

    res = 0
    for i in range(m):
        sub = []
        for j in range(1,m):
            fila = []
            for k in range(m):
                if k != i:
                    fila.append(A[j][k])
                sub.append(fila)
            menor = det(sub)
            signo = (-1)**i
            res += signo*A[0][i]*menor
    return res  


def inversa(A):
    A = np.array(A, dtype=np.float64)
    m, n = A.shape 

    if m != n:
        return None 

    if abs(det(A)) < 1e-9: # no se que tol elegir 1e-9 o 1e-15
        return None 

    A_inv = np.eye(m)
    A_copia = A.copy() 

    for i in range(m):
        if abs(A_copia[i][i]) < 1e-15:
            obs = False
            for j in range(i+1,m):
                if abs(A_copia[j][i]) > 1e-15:
                    A_copia[[i, j]] = A_copia[[j, i]]
                    A_inv[[i, j]] = A_inv[[j, i]]
                    obs = True 
                    break 
            if not obs:
                return None 


        pivote = A_copia[i,i]
        A_copia[i] = A_copia[i] / pivote 
        A_inv[i] = A_inv[i] / pivote 

        for j in range(m):
            if j != i:
                coef = A_copia[j,i]
                A_copia[j] = A_copia[j] - coef*A_copia[i]
                A_inv[j] = A_inv[j] - coef*A_inv[i]
    return A_inv 

#%% calcula LDV 

def diagonal(A):
    A_copia = A.copy()
    m, n = A.shape 
    B = np.zeros((m, m))

    for i in range(m):
        pivote = A_copia[i,i]

        if abs(pivote) < 1e-12:
            return None 

        B[i,i] = pivote 

        for j in range(i+1, m):
            factor = A_copia[j,i] / pivote 
            for k in range(i, m):
                A_copia[j,k] -= factor * A_copia[i,k]
    return B 

def calculaLDV(A): # pedia que devuelva nops tamb pero los test no, asi q ni idea
    A = np.array(A, dtype=np.float64)
    m,n = A.shape

    L, U, nops = calculaLU(A) 
    if (L is None or U is None) and (m != n):
        return None, None, None #nops 

    D = diagonal(U)
    V = U.copy() 

    for i in range(m):
        if abs(V[i,i]) > 1e-08:
            div = V[i,i]
            for j in range(n):
                V[i,j] = V[i,j] / div 
                nops += 1 
        else:
            return None, None, None #nops

    return L, D, V # nops


#%% es SDP??

def esSDP(A, atol=1e-8):
    A = np.array(A, dtype=np.float64)
    m, n = A.shape 

    if m != n:
        return False 

    for i in range(m):
        for j in range(i+1, m):
            if abs(A[i,j]-A[j,i]) > atol:
                return False 

    L, D, V =  calculaLDV(A) # le saque los nops 
    if L is None or D is None or V is None:
        return False 

    D = np.array(D, dtype=np.float64) 
    d, c = D.shape 

    for i in range(d):
        if D[i,i] <= atol:
            return False 

    return True 
    

#%% cholesky (me falta hacer esta) 

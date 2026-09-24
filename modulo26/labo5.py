import numpy as np 
from labo3 import norma 

def producto(x, y, interno=True): # funcion que devuelve producto externo o interno 
    if interno:
        res = 0 
        for i in range(len(x)):
            res += x[i]*y[i] 
        return res 
    else:
        res = []
        for i in range(len(x)):
            fila = []
            for j in range(len(y)):
                fila.append(x[i]*y[j])
            res.append(fila)
        return np.array(res, dtype=np.float64) # qcy si es necesario esto lo puse en todos 

def QR_con_GS(A, tol = 1e-12, retorna_nops = False):
    A = np.array(A, dtype=float) 
    m,n = A.shape 
    
    if m != n:
        if retorna_nops:
            return None, None, None 
        return None, None 
    
    nops = 0 
    Q = np.zeros((m,n), dtype=float)
    R = np.zeros((n,n), dtype=float)
    
    r11 = norma(A[:, 0], 2)
    nops += 2 *n 
    
    if abs(r11) < tol:
        Q[:, 0] = 0 
    else: 
        Q[:, 0] = A[:, 0] / r11 
        nops += n 
    
    R[0,0] = r11 
    
    for j in range(1, n):
        q_ = A[:, j].copy() 
        nops += n 
        
        for k in range(j):
            r_kj = producto(Q[:, k], A[:, j], True)
            nops += 2 * n 
            
            R[k, j] = r_kj 
            
            q_ = q_ - r_kj * Q[:, k]
            nops += 2 * n 
        
        r_jj = norma(q_, 2)
        nops += 2 * n
        
        if abs(r_jj) < tol:
            Q[:, j] = 0 
        else: 
            Q[:, j] = q_ / r_jj
            nops += n 
        
        R[j, j] = r_jj 
    
    if retorna_nops:
        return Q, R, nops 
    else: 
        return Q, R 
            
        
def QR_con_HH(A, tol=1e-12):
    A = np.array(A, dtype=np.float64)
    m, n = A.shape 
    
    R = A.copy().astype(np.float64)
    Q = np.eye(m)
    
    for k in range(min(m,n)):
        x = R[k:, k] 
        
        if norma(x,2) < tol:
            continue 
        
        alpha = norma(x,2)
        if x[0] >= 0:
            alpha = -alpha 
        
        u = x.copy() 
        u[0] = u[0] - alpha 
        u_norm = norma(u,2)
        
        if u_norm < tol:
            continue 
        
        u = u / u_norm 
    
        H_k = np.eye(len(u)) - 2 * producto(u, u, False)
        
        R[k:, k:] = H_k @ R[k:, k:]
        
        Q[:, k:] = Q[:, k:] @ (H_k).T 
        
    return Q, R


def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH' or len(A) == len(A[0]): 
        return QR_con_HH(A, tol) 
    else:
        return QR_con_GS(A, tol) 


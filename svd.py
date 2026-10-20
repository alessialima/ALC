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


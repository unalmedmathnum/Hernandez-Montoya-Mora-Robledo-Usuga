import numpy as np

def power_method(A:np.matrix, max_iter=100, tolerance=1e-6):
    A = np.array(A.tolist())

    n = A.shape[0]

    # Vector inicial aleatorio
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    eigenvalue_prev = 0

    for _ in range(max_iter):
        # Multiplicación de matriz por vector
        Av = A @ v

        # Calcular nuevo valor propio
        eigenvalue = np.dot(v, Av)

        # Normalizar vector
        v_new = Av / np.linalg.norm(Av)

        # Criterio de convergencia
        if np.abs(eigenvalue - eigenvalue_prev) < tolerance:
            return eigenvalue, v_new

        v = v_new
        eigenvalue_prev = eigenvalue

    return eigenvalue, v



import numpy as np

def power_method(A, max_iter=100, tolerance=1e-6):
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




A = np.array([[2, 1],
              [3, 4]])

B = np.array([[3, 2],
              [3, 4]])

C = np.array([[2, 3],
              [1, 4]])

D = np.array([[1, 1, 2],
              [2, 1, 1],
              [1, 1, 3]])

E = np.array([[1, 1, 2],
              [2, 1, 3],
              [1, 1, 1]])

F = np.array([[2, 1, 2],
              [1, 1, 3],
              [1, 1, 1]])

G = np.array([[1, 1, 1, 2],
              [2, 1, 1, 1],
              [3, 2, 1, 2],
              [2, 1, 1, 4]])

H = np.array([[1, 2, 1, 2],
              [2, 1, 1, 1],
              [3, 2, 1, 2],
              [2, 1, 1, 4]])
x0 = np.array([1, 1])
List= [A,B,C,D,E,F,G,H]
c=1
for i in List:
  print("--"*9,c,"--"*9);c+=1
  print(i)
  eigenvalue, eigenvector = power_method(i)
  print("valor propio dominante:", eigenvalue)
  print("vector propio correspondiente:", eigenvector)
  eigenvalues, eigenvectors = np.linalg.eig(i)
  print("\nValores propios de numpy:", eigenvalues)
  print("Vector propio numpy para mayor valor propio:", eigenvectors[:, np.argmax(eigenvalues)])

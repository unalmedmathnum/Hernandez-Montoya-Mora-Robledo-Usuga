class QRDecomposition :
    def __init__(self, A, max_iter=1000, tol=1e-10, method=0, use_shift=False, plot_log=True, use_hessenberg=True) -> None:
        """
        Inicializa los parámetros para realizar la descomposición QR.

        Parámetros:
        - A (numpy.ndarray): Matriz cuadrada de entrada.
        - max_iter (int): Número máximo de iteraciones.
        - tol (float): Tolerancia para determinar la convergencia.
        - method (int): Método para QR (0: Gram-Schmidt, 1: Householder).
        - use_shift (bool): Activar o desactivar desplazamiento espectral.
        - plot_log (bool): Si True, la gráfica de pérdida usa escala logarítmica.
        """
        
        self.A = A.astype(float)  # Convertir la matriz A a tipo float
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []  # Historial de pérdida (norma fuera de la diagonal)
        self.method = method  # Método QR seleccionado
        self.use_shift = use_shift  # Desplazamiento espectral activado/desactivado
        self.plot_log = plot_log  # Usar escala logarítmica en la gráfica
        self.use_hessenberg = use_hessenberg # Usar la transformación de Hessenberg

    def main(self):
        """
        Ejecuta el método QR iterativo para aproximar los valores propios de una matriz.

        Retorna:
        list: Lista con los valores propios aproximados.
        """
        import numpy as np

        # Transformar la matriz A a forma de Hessenberg antes del proceso QR
        if self.use_hessenberg :
            Ak = self.transform_to_hessenberg()
        else :
            Ak = self.A.copy()  # Copia de la matriz A para no modificar la original

        n = self.A.shape[0]
        
        for i in range(self.max_iter):
            # Desplazamiento espectral
            if self.use_shift:
                # Si la pérdida aumenta, desactiva el desplazamiento espectral
                if i > 1 and self.loss_history[-1] > self.loss_history[-2]:
                    mu = 0
                else:
                    # Selección de desplazamiento espectral
                    # mu = Ak[-1, -1]  # Diagonal inferior derecha
                    mu = Ak.diagonal().max()  # Máximo de la diagonal
                    Ak -= mu * np.eye(n)  # Restar mu * I
            else:
                mu = 0  # Sin desplazamiento

            # Descomposición QR según el método seleccionado
            if self.method == 0:
                Q, R = self.qr_factorization_gram_schmidt(Ak)
            elif self.method == 1:
                Q, R = self.qr_factorization_householder(Ak)
            else:
                raise ValueError("Método inválido. Usa 'gram-schmidt' o 'householder'.")

            # Actualizar la matriz A
            Ak = R @ Q + mu * np.eye(n)  # Restaurar desplazamiento espectral si se aplicó

            # Calcular la norma fuera de la diagonal (pérdida)
            off_diagonal_norm = np.sqrt(np.sum(np.triu(Ak, k=1)**2))
            self.loss_history.append(off_diagonal_norm)

            # Verificar convergencia
            if off_diagonal_norm < self.tol:
                self.plot_loss()
                eigenvalues = np.diag(Ak)  # Extraer los valores propios aproximados
                return eigenvalues.tolist()
        
        # Si no converge, graficar la pérdida y lanzar excepción
        self.plot_loss()
        raise Exception("ERROR: El método no converge para esta matriz.")

    def qr_factorization_gram_schmidt(self, A):
        """
        Realiza la descomposición QR usando el método de Gram-Schmidt.

        Parámetros:
        A (numpy.ndarray): Matriz cuadrada de entrada.

        Retorna:
        tuple: Matrices Q (ortogonal) y R (triangular superior).
        """
        import numpy as np

        n = A.shape[0]
        Q = np.zeros_like(A, dtype=float)  # Inicializa Q
        R = np.zeros((n, n), dtype=float)  # Inicializa R
        
        for j in range(n):
            v = A[:, j].astype(float).copy()  # Copiar la columna actual
            for i in range(j):
                # Proyección de la columna actual sobre las columnas anteriores
                R[i, j] = np.dot(Q[:, i].A1, A[:, j].A1)
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)  # Calcular la norma

            if R[j, j] > 1e-10:  # Evitar división por cero
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = 0  # Si no es posible normalizar, asignar ceros
    
        return Q, R

    def qr_factorization_householder(self, A):
        """
        Realiza la descomposición QR usando reflexiones de Householder.

        Parámetros:
        A (numpy.ndarray): Matriz cuadrada de entrada.

        Retorna:
        tuple: Matrices Q (ortogonal) y R (triangular superior).
        """
        import numpy as np

        n = A.shape[0]
        Q = np.eye(n)  # Matriz identidad para construir Q
        R = A.copy()

        for k in range(n - 1):
            # Crear el vector de Householder
            x = R[k:, k]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x)
            v = x - e1
            norm_v = np.linalg.norm(v)

            # Evitar división por cero
            if norm_v > 1e-10:  # Si la norma es suficientemente grande
                v /= norm_v  # Normalizar v
            else:
                v = np.zeros_like(v)  # Si no, asignar ceros a v

            # Construir la matriz de reflexión H
            H_k = np.eye(n)
            H_k[k:, k:] -= 2.0 * np.outer(v, v)

            # Actualizar R y Q
            R = H_k @ R
            Q = Q @ H_k.T

        return Q, R
        
    def transform_to_hessenberg(self):
        """
        Transforma la matriz A a su forma de Hessenberg usando
reflexiones de Householder.

        Retorna:
        numpy.ndarray: Matriz en forma de Hessenberg.
        """
        import numpy as np

        n = self.A.shape[0]
        H = self.A.copy()  # Copia de la matriz para no modificar la original

        for k in range(n - 2):
            # Crear el vector de Householder para la columna k
            x = H[k+1:, k]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x)
            v = x - e1
            norm_v = np.linalg.norm(v)

            if norm_v > 1e-10:  # Evitar división por cero
                v /= norm_v
            else:
                continue  # Saltar si no se necesita la reflexión

            # Construir la matriz de reflexión
            H_k = np.eye(n)
            H_k[k+1:, k+1:] -= 2.0 * np.outer(v, v)

            # Aplicar la transformación a la matriz H
            H = H_k @ H @ H_k.T

        return H

    def plot_loss(self):
        """
        Genera una gráfica de la pérdida (norma fuera de la diagonal) a lo largo de las iteraciones.
        """
        import matplotlib.pyplot as plt

        if not self.loss_history:
            raise Exception("No hay datos de pérdida disponibles. Ejecuta 'main' primero.")
        
        print(min(self.loss_history))  # Imprimir el valor mínimo de la pérdida

        plt.figure(figsize=(8, 6))
        plt.plot(self.loss_history, marker='o', linestyle='-')
        plt.title("Pérdida (Norma fuera de la diagonal) vs Iteraciones")
        plt.xlabel("Iteraciones")
        plt.ylabel("Pérdida (Norma)")
        if self.plot_log:  # Escala logarítmica opcional
            plt.yscale('log')
        plt.grid(True)
        plt.show()

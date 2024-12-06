import numpy as np

def power_method(A:np.matrix, max_iter=100, tolerance=1e-6):
    A = np.array(A.tolist())

    n = A.shape[0]

    loss_history = []

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

        loss_history.append(np.abs(eigenvalue - eigenvalue_prev))

        # Criterio de convergencia
        if np.abs(eigenvalue - eigenvalue_prev) < tolerance:
            plot_loss(loss_history)
            return eigenvalue, v_new

        v = v_new
        eigenvalue_prev = eigenvalue

    plot_loss(loss_history)
    return eigenvalue, v

def plot_loss(loss_history: list):
    """
    Genera una gráfica de la pérdida (norma fuera de la diagonal) a lo largo de las iteraciones.
    """
    import matplotlib.pyplot as plt

    if not loss_history:
        raise Exception("No hay datos de pérdida disponibles. Ejecuta 'main' primero.")
    
    #print(min(self.loss_history))  # Imprimir el valor mínimo de la pérdida

    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, marker='o', linestyle='-')
    plt.title("Pérdida (Diferencia valores propios) vs Iteraciones")
    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida (Norma)")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

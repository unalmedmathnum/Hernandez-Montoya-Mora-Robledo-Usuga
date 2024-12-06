import numpy as np
from sympy import symbols, Matrix, pretty, diff, div, factor


# Calculamos simbólicamente el polinomio característico de la matriz y se retorna para aplicarlo en la búsqueda de raíces

def calcular_polinomio_caracteristico(elementos: np.matrix):

    elementos = elementos.tolist()

    # Ingrese una matriz A cuadrada de nxn en formato lista de filas : A = [[fila_1],[fila_2],...,[fila_n]]

    # Verificar que la matriz sea cuadrada
    if not all(len(fila) == len(elementos) for fila in elementos):
        raise ValueError("La matriz debe ser cuadrada.")

    # Crear la matriz simbólica
    matriz = Matrix(elementos)
    print("\nMatriz ingresada (redondeada):")
    print(matriz.evalf(3))

    # Crear el símbolo lambda para el polinomio característico
    lambda_symbol = symbols('lambda')

    # Calcular la matriz característica (A - lambda*I)
    identidad = Matrix.eye(len(elementos))
    matriz_caracteristica = matriz - lambda_symbol * identidad

    # Calcular el determinante para obtener el polinomio característico
    polinomio_caracteristico = matriz_caracteristica.det()
    print("\nEl polinomio característico de la matriz es:")
    print(pretty(polinomio_caracteristico, use_unicode=True))
    return polinomio_caracteristico



# Usamos un método integrado de bisección y Newton para buscar raíces del polinomio característico que corresponden a valores propios de la matriz
# Esencialmente buscamos cambios de signo y aplicamos Newton para intentar hallar rápidamente una raíz en el intervalo dado, si no funciona, aplicamos bisección y repetimos el proceso
def encontrar_raices_biseccion_newton(polinomio, a, b, iter_Newton=10, max_iter_biseccion=20, tolerancia=1e-10):

    # Parámetros de entrada al algoritmo de búsqueda de raíces de polinomios
    # polinomio: Polinomio a encontrar raíces, pase como parámetro el método calcular_polinomio_caracteristico(A) que retorna el polinomio de la matriz A 
    # a,b : Los extremos del intervalo donde se buscarán raíces.
    # iter_Newton : Parámetro para indicar el número de iteraciones máximas que hará el método de Newton para intentar hallar una raíz
    # max_iter_biseccion : Número máximo de bisecciones que hará el algoritmo para hallar raíces en el intervalo
    # tolerancia : Cota para establecer si un x es una raíz del polinomio, corresponde a la distancia |f(x)-0|= |f(x)| de aceptación de raíces

    # Verificación de intervalo no degenerados
    if a >= b:
      print("Error en los parámetros de entrada, verifique el intervalo de búsqueda")
      return None

    # Crear el símbolo lambda
    lambda_symbol = symbols('lambda')

    # Definición de la función polinómica y su derivada
    f = polinomio
    f_derivada = diff(f, lambda_symbol)

    # Inicializar el contador de raíces
    grado = f.as_poly().degree()
    contador_raices = 0

    # Conjunto de las raíces encontradas
    raices_encontradas = set()

    # Intervalo de partida
    intervalo_inicio = a
    intervalo_fin = b

    contador_biseccion = 0

    # Mientras no hayamos encontrado todas las raíces o alcancemos el límite de iteraciones de bisección
    while (len(raices_encontradas) < grado) and (contador_biseccion < max_iter_biseccion) :

        x = (intervalo_inicio + intervalo_fin) / 2  # El medio del intervalo

        # Evaluar f en los extremos del intervalo y verificar si son raíces
        # Si encuentra una raíz, bisecta y actualiza contadores
        f_inicio = f.evalf(subs={lambda_symbol: intervalo_inicio})
        if abs(f_inicio) <= tolerancia:
          if intervalo_inicio not in raices_encontradas: # Verifica si la raíz ya fue añadida al conjunto
            raices_encontradas.add(intervalo_inicio)
            contador_raices += 1
          intervalo_inicio = x
          contador_biseccion += 1
          continue

        f_fin = f.evalf(subs={lambda_symbol: intervalo_fin})
        if abs(f_fin) <= tolerancia:
          if intervalo_fin not in raices_encontradas:
            raices_encontradas.add(intervalo_fin)
            contador_raices += 1
          intervalo_fin = x
          contador_biseccion += 1
          continue


        # Si los extremos no son raíces, evaluamos el medio
        f_x = f.evalf(subs={lambda_symbol: x})  # Evalúa f en el punto inicial y mira si es raíz aproximada
        if abs(f_x)<= tolerancia: # Bisectamos por la derecha para continuar el método
          if x not in raices_encontradas:
            raices_encontradas.add(x)
            contador_raices += 1
          contador_biseccion += 1
          intervalo_fin = (intervalo_inicio + x) / 2
          continue

        # Si aún no encuentra raíces, comprueba si hay un cambio de signo en el intervalo
        if f_inicio * f_fin < 0:

            # Sabemos que hay al menos una raíz, ahora intentamos usar Newton para hallarla rápidamente

            # Aplicamos el método de Newton una cantidad fija de veces para intentar hallar una raíz con la tolerancia dada

            for i in range(iter_Newton):  # Si el medio no era raíz, itera Newton la cantidad de veces indicada

                f_x_derivada = f_derivada.evalf(subs={lambda_symbol: x})  # Evalúa la derivada de f en el punto medio
                if f_x_derivada == 0:   # Verificamos que la derivada no se anule
                    contador_biseccion += 1  # Bisectamos otra vez para repetir el proceso
                    intervalo_fin = (intervalo_inicio + intervalo_fin) / 2
                    continue

                x_nuevo = x - f_x / f_x_derivada  # El nuevo valor de x dado por el método de Newton

                # Si encontramos una raíz aproximada, bisectamos para repetir el proceso
                if abs(f.evalf(subs={lambda_symbol: x_nuevo})) <= tolerancia:
                    if  x_nuevo not in raices_encontradas:
                        raices_encontradas.add(x_nuevo)
                        contador_raices += 1
                    contador_biseccion += 1
                    intervalo_fin = (intervalo_inicio + intervalo_fin) / 2
                    continue

                # Actualizar x
                x = x_nuevo

            # Si Newton no encuentra una raíz aproximada, aún podemos encontrarla bisectando otra vez
            # Tenemos que elegir un lado de bisección, comprobamos cuál de ellos tiene cambio de signo
            if (f_x * f_inicio < 0): # Hay una raíz en la parte izquierda
              contador_biseccion += 1
              intervalo_fin = (intervalo_inicio + intervalo_fin) / 2
              continue
            else: # La raíz está a la derecha
                contador_biseccion += 1
                intervalo_inicio = (intervalo_inicio + intervalo_fin) / 2
                continue

        else:
            # Bisectamos y continuamos con el subintervalo izquierdo
            contador_biseccion += 1
            intervalo_fin = (intervalo_inicio + intervalo_fin) / 2


    print("\nRaíces encontradas = Valores propios encontrados:", raices_encontradas)
    return raices_encontradas


# Ejemplo de uso
#matriz = [[1,1,2], [2,1,1],[1,1,3]]
#raices = encontrar_raices_biseccion_newton(calcular_polinomio_caracteristico(matriz), a=-10, b=100, iter_Newton=5, max_iter_biseccion=10, tolerancia=0.1)




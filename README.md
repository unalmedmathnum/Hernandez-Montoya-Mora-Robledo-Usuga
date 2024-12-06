# Métodos para Cálculo de Valores Propios

## Integrantes
- Camilo Montoya Arango
- Juan Pablo Robledo Meza
- Mateo Sebastián Mora Montero
- Pablo Andrés Úsuga Gomez
- Daniel Andrés Hernández Pedraza

Este proyecto implementa tres métodos distintos para calcular los valores propios de matrices:

1. **Polinomio Característico**  
2. **Método de la Potencia (Power Method)**  
3. **Descomposición QR (QR Decomposition)**

## Descripción del Problema

Los valores propios son fundamentales en diversas áreas de matemáticas, ingeniería y ciencias. Son esenciales para resolver sistemas de ecuaciones diferenciales, realizar reducción de dimensionalidad (por ejemplo, en PCA) y analizar la estabilidad de sistemas, entre otras aplicaciones.  

Este proyecto proporciona implementaciones en Python de tres métodos comunes para calcular valores propios, cada uno con sus propias ventajas y consideraciones computacionales.

## Instrucciones de Uso

El código para cada método está ubicado en el directorio `src`. A continuación, se describen las instrucciones para utilizar cada implementación:

### 1. Polinomio Característico  
Este método calcula los valores propios resolviendo la ecuación característica, \( \det(A - \lambda I) = 0 \), utilizando una combinación del método de Bisección y el Método de Newton para buscar las raíces del polinomio característico.

- **Función Principal:**  
  `encontrar_raices_biseccion_newton`

- **Descripción de la Función:**  
  Esta función encuentra los valores propios (raíces del polinomio característico) de una matriz \( A \) en un intervalo dado. Combina iterativamente el método de Bisección y el Método de Newton, asegurando alta precisión en los resultados.  

  **Parámetros:**  
  - `A`: Matriz cuadrada para la cual se calcularán los valores propios.
  - `a`, `b`: Extremos del intervalo donde se buscarán las raíces.
  - `iter_Newton`: Máximo número de iteraciones del Método de Newton por intento (por defecto 10).
  - `max_iter_biseccion`: Máximo número de particiones por bisección (por defecto 20).
  - `tolerancia`: Precisión requerida para aceptar un valor como raíz (por defecto \( 10^{-10} \)).

- **Uso:**  
  ```python
  from src.poly_car import *

  import numpy as np

  matriz = np.array([[2, -1], [-1, 2]])
  intervalo_inicio = -10
  intervalo_fin = 10

  valores_propios = encontrar_raices_biseccion_newton(
      matriz, intervalo_inicio, intervalo_fin, iter_Newton=15, max_iter_biseccion=25, tolerancia=1e-8
  )
  print("Valores propios:", valores_propios)

### 2. Método de la Potencia  
El Método de la Potencia es un algoritmo iterativo para encontrar el valor propio dominante (el mayor en magnitud) y su vector propio correspondiente de una matriz \( A \).

- **Función Principal:**  
  `power_method`

- **Descripción de la Función:**  
  Esta función calcula el valor propio dominante y su vector propio de una matriz \( A \) utilizando un enfoque iterativo. El método normaliza un vector inicial aleatorio a través de sucesivas multiplicaciones con la matriz hasta que el valor propio converge.

  **Parámetros:**  
  - `A`: Matriz cuadrada para la cual se calculará el valor propio dominante.
  - `max_iter`: Máximo número de iteraciones permitidas (por defecto 100).
  - `tolerance`: Precisión requerida para aceptar la convergencia del valor propio (por defecto \( 10^{-6} \)).

  **Salida:**  
  - `eigenvalue`: Valor propio dominante aproximado.
  - `v`: Vector propio asociado al valor propio dominante.

- **Uso:**  
  ```python
  from src.powermethod import *

  import numpy as np

  matriz = np.matrix([[4, 1], [2, 3]])

  valor_propio, vector_propio = power_method(matriz, max_iter=200, tolerance=1e-8)

  print("Valor propio dominante:", valor_propio)
  print("Vector propio asociado:", vector_propio)

### 3. Descomposición QR  
La Descomposición QR es un método iterativo basado en factorizaciones QR sucesivas para calcular todos los valores propios de una matriz \( A \). Este método puede incluir optimizaciones como la transformación de Hessenberg y el desplazamiento espectral.

- **Clase Principal:**  
  `QRDecomposition`

- **Descripción de la Clase:**  
  La clase `QRDecomposition` implementa un método iterativo para aproximar los valores propios de una matriz cuadrada. Permite elegir entre diferentes técnicas de factorización QR y ofrece opciones para optimizar la convergencia, como el uso de la transformación de Hessenberg y el desplazamiento espectral.

  **Parámetros:**  
  - `A`: Matriz cuadrada para la cual se calcularán los valores propios.
  - `max_iter`: Número máximo de iteraciones permitidas (por defecto 1000).
  - `tol`: Precisión requerida para aceptar la convergencia del método (por defecto \( 10^{-10} \)).
  - `method`: Método de factorización QR (0: Gram-Schmidt, 1: Householder).
  - `use_shift`: Activar o desactivar el desplazamiento espectral (por defecto `True`).
  - `plot_log`: Si es `True`, utiliza escala logarítmica para graficar la pérdida.
  - `use_hessenberg`: Activar o desactivar la transformación de Hessenberg antes de aplicar QR (por defecto `True`).

  **Método Principal:**  
  - `main()`: Ejecuta el método iterativo para calcular los valores propios de la matriz y retorna una lista de valores propios aproximados.

- **Uso:**  
  ```python
  from src.qrDecomposition import QRDecomposition
  import numpy as np

  # Crear una matriz cuadrada
  matriz = np.array([[5, 4, 2],
                     [1, 3, 7],
                     [6, 0, 9]])

  # Inicializar el método QR
  qr_solver = QRDecomposition(A=matriz, max_iter=500, tol=1e-8, method=1, use_shift=True)

  # Calcular valores propios
  valores_propios = qr_solver.main()

  print("Valores propios aproximados:", valores_propios)

## Experimentos

Los experimentos realizados con los tres métodos implementados se encuentran documentados en el archivo **`experimentos.ipynb`**, disponible en el repositorio.

Este notebook incluye:
- Ejemplos prácticos del uso de cada método.
- Comparaciones entre los métodos en términos de precisión y convergencia.
- Visualizaciones del proceso iterativo, como el historial de pérdida en el Método de la Potencia y la Descomposición QR.


![](https://repository-images.githubusercontent.com/899298969/8073b1ae-2cff-403d-a217-efff5c5405e2)

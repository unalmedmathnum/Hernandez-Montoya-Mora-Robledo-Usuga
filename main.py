import numpy as np
from src.poly_car import *

A = np.matrix([[2,1],[3,4]])
encontrar_raices_biseccion_newton(calcular_polinomio_caracteristico(A), 0, 5)

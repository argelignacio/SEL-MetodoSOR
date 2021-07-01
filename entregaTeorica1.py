import numpy as np
import matplotlib.pyplot as plt
from sympy import *


def sorHilbert(N,w,semilla):

    print("Se ca a calcular la solucion por metodo SOR con N=" +str(N)+ " un w=" +str(w)+ " y una semilla = "+ str(semilla)+".")

    'constantes a usar en el problema'
    a = 0
    b = 0.05 + 103807*(10**(-5))
    'Defino la matriz de hilbert'
    arreglos =[]
    for j in range(N):
        subarreglo = [0 for i in range(N)]
        for k in range(N):
            constanteHilbert = j+k+1
            subarreglo[k] = (b**(constanteHilbert) - a**(constanteHilbert))/constanteHilbert
        arreglos.append(subarreglo)
    hilbertMatrix = np.array(arreglos)
    

    'defino el termino independiente'
    x= Symbol('x')
    arregloIndep = []
    for j in range(N):
        fx = -4*x*(x-1)*(x**j)
        res = Integral(fx, (x, a, b)).doit()
        arregloIndep.append(res) 
    c = np.array(arregloIndep)

    'defino L, U, D'

    U = hilbertMatrix.copy()
    for fil in range(N):
        for col in range(N):
            if (col <= fil) :
                U[fil,col] = 0

    L = hilbertMatrix.copy()
    for fil in range(N):
        for col in range(N):
            if (fil <= col) :
                L[fil,col] = 0


    D = hilbertMatrix.copy()
    for fil in range(N):
        for col in range(N):
            if (fil != col) :
                D[fil,col] = 0

    U = U * -1
    L = L * -1
    
    'calculo Tw'
    Tw = np.linalg.inv(D - w*L) @ ((1-w) * D+ w * U)
    'calculo Cw'
    Cw = (w * np.linalg.inv(D - w*L)) @ c
    
    'Calculo SOR'
    
    iteraciones = 0
    parada = 10 ** (-4)
    tol = 1 
    x_k = semilla
    while((tol > parada)):
        x_kmenos1 = x_k.copy()
        x_k = Tw @ x_k + Cw
        diferencia = x_k - x_kmenos1
        iteraciones +=1
        tol = (np.linalg.norm(diferencia, np.inf))/(np.linalg.norm(x_k, np.inf))

    plt.stem(w, iteraciones)
    print('w usado:\n')
    print(w)
    print('Vector resultante:\n')
    print(x_k)
    print('Iteraciones hasta llegar a la tolerancia')
    print(iteraciones)


N=10
semilla = np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0])
sorHilbert(N,1.1,semilla)
sorHilbert(N,1.2,semilla)
sorHilbert(N,1.3,semilla)
sorHilbert(N,1.4,semilla)
act = 1.5
for i in range(10): #Sabiendo que Woptimo esta en este rango, iteramos de forma mas precisa
    act += 0.02
    sorHilbert(N,act,semilla)
sorHilbert(N,1.7,semilla)
sorHilbert(N,1.8,semilla)
sorHilbert(N,1.9,semilla)
plt.title('Comparativa')
plt.xlabel('w')
plt.ylabel('Iteraciones')
plt.show()
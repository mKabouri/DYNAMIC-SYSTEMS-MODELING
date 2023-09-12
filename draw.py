import matplotlib.pyplot as plt
import numpy as np

from method import circle, disp_2D, disp_graph, meth_epsilon
from heun import step_heun
from method import arctan_prime
import method as m

def show_tangent_field1D(t0, tf, h, F , meth):
    
    K = int((tf - t0)/h)
    a = t0
    b = tf
    N = K * 1j

    Y, X = np.mgrid[a:b:N, a:b:N]
    U = np.zeros((K, K))
    V = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            U[i][j] = t0 + h
            V[i][j] =  meth([Y[i][j]], X[i][j], h, F)
    normes = np.sqrt(U**2 + V**2)
    U = -U
    UU = U/normes
    VV = V/normes
    plt.axis([a, b, a, b])

    plt.quiver(X, Y, UU, VV, headlength=5, width=0.002)
    values = np.linspace(-0.5, 0.5, 5)
    for i in range(len(values)):
        Y1D, X1D = m.meth_n_step(np.array([values[i]]), -2, K, h, arctan_prime, step_heun)
        disp_graph(X1D, Y1D)
    plt.show()

def show_tangent_field2D(t0, tf, h, F, meth):
    K = int((tf - t0)/h)
    a = t0
    b = tf
    N = K * 1j
    Y, X = np.mgrid[a:b:N, a:b:N]
    U = np.zeros((K, K))
    V = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            tab = meth([Y[i][j], X[i][j]], t0, h, F)
            U[i][j] = tab[0]
            V[i][j] =  tab[1]
    normes = np.sqrt(U**2 + V**2)
    U = -U
    UU = U/normes
    VV = V/normes
    plt.quiver(X, Y, UU, VV, headlength=5, width=0.002)
    values = np.linspace(-2, -0.2, 5)
    for i in range(len(values)):
        Y2D, X2D = m.meth_n_step(np.array([values[i], 0]), 0, K, 6.25/K, circle, step_heun)
        disp_2D(Y2D)
    plt.show()

y0 = np.array([1,1])
t0 = -2
tf = 2
eps = 0.1
def F(y1, y2) :
    res = [-y2, y1]
    print(-y2)
    print('-----------')
    print(y1)
    print('-----------')
    print(res)
    return -y2, y1
    
show_tangent_field1D(t0, tf, eps, arctan_prime, step_heun)
show_tangent_field2D(-2, 2, eps, circle, step_heun)
'''
import matplotlib.pyplot as plt
import numpy as np
K=25
a=-2
b=2
N=K*1j

Y, X = np.mgrid[a:b:N, a:b:N]

U = -Y
V = X
normes = np.sqrt(U**2 + V**2)
UU = U/normes
VV = V/normes
plt.axis([a, b, a, b])

plt.quiver(X, Y, UU, VV,headlength=5, width=0.002)

plt.show()
'''
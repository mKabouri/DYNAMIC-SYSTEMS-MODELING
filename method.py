import heun as he
import mat as m
import euler as e
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

def norm_inf(y2N, yN, N):
    res = -1
    for i in range(2*N):
        if (i%2 == 0):
            x_i = np.linalg.norm(y2N[i] - yN[i//2])
            if (x_i > res):
                res = x_i
    return res

def meth_n_step(y0, t0, N, h, f, meth) :
    yN = [y0]
    t = [t0]
    for i in range(N):
        yN.append(meth(yN[-1], t[-1], h, f))
        t.append(t[len(t)-1] + h)
    return np.array(yN), np.array(t)

def meth_epsilon(y0, t0, tf, eps, f , meth):
    N = 2
    h = (tf - t0)/N
    yN, tN = meth_n_step(y0, t0, N, h, f, meth)
    y2N, t2N = meth_n_step(y0, t0, 2*N, h/2, f, meth)
    while( norm_inf(y2N, yN, N) > eps ) :
        h /= 2
        N *= 2
        yN, tN = cp.deepcopy(y2N), cp.deepcopy(t2N)
        y2N, t2N = meth_n_step(y0, t0, 2*N, h/2, f, meth)
    return np.array(y2N), np.array(t2N)


def arctan_prime(y, t):
    return np.array([y[0]/(1+t**2)])

def circle(y, t):
    return np.array([-y[1], y[0]])

N = 100
#y2d1, t2d1 = meth_n_step(np.array([1, 0]), 0, N, 10/N, circle, e.step_euler)
#y2d2 , t2d2 = meth_epsilon(np.array([2, 0]), 0, 100, 10**(-2), circle, e.step_euler)
#Y1D, X1D = meth_n_step(np.array([1]), 0, N, 10/N, arctan_prime, e.step_euler)
#Yepsilon, Xepsilon = meth_epsilon([1], 0, 1, 10**(-3), arctan_prime, e.step_euler)
def y1(t):
    return np.cos(t) + np.sin(t)
def y2(t):
    return np.sin(t) - np.cos(t)
values = np.linspace(-10, 10, 1000)
X_test = [y1(values[i]) for i in range(1000)]
Y_test = [y2(values[i]) for i in range(1000)]
# print(X_test, Y_test)

def disp_graph(X, Y):
    plt.plot(X, Y)

def disp_2D(y):
    n = len(y)
    X = [y[i][0] for i in range(n)]
    Y = [y[i][1] for i in range(n)]
    plt.plot(X,Y)    


# disp_graph(X_test, Y_test)
#X2D1 = [y2d1[i][0] for i in range(len(y2d1))]
#Y2D1 = [y2d1[i][1] for i in range(len(y2d1))]
#print("y2d2= ", y2d2)
#disp_2D(y2d2)

#disp_graph(X_test, Y_test)
#disp_graph(X2D1, Y2D1)
#plt.show()
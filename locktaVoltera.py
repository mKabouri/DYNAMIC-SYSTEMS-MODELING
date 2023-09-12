import numpy as np
import matplotlib.pyplot as plt

from method import *
from heun import step_heun



def premier_système() :
    def dN(y, t) :
        return gamma * y * (1 - y/k)
    
    gamma = 0.1
    k = 3

    y0 = 10
    t0 = 0
    tf = 100
    eps = 0.1

    yN, tN = meth_epsilon(y0, t0, tf, eps, dN, step_heun)

    plt.plot(tN, yN)
    plt.show()


def deuxieme_systeme() :
    a, b, c, d = 4, 3, 2, 1
    
    def LV(y, t) :
        dN = y[0] * (a - b * y[1])
        dP = y[1] * (c * y[0] - d)
        return dN, dP
    
    y0 = np.array([100, 10])
    t0 = 0
    tf = 100
    eps = 0.1
    
    yN, tN = meth_epsilon(y0, t0, tf, eps, LV, step_heun)
    
    plt.plot(tN, yN[0], yN[1])
    plt.show()
    
    
deuxieme_systeme()
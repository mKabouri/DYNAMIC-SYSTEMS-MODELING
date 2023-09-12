import numpy as np

def step_euler(yn, tn, h, f):
    return yn + h*f(yn, tn)

def step_runge_kutta4(yn, tn, h, f):
    pn1 = f(yn, tn)
    yn2 = yn + (1/2)*h*pn1
    pn2 = f(yn2, tn + (1/2)*h)
    yn3 = yn + (1/2)*h*pn2
    pn3 = f(yn3, tn + (1/2)*h)
    yn4 = yn + h*pn3
    pn4 = f(yn4, tn + h)
    return yn + (1/6)*h*(pn1 + 2*pn2 + 2*pn3 + pn4)



def step_heun(y, t, h, f) :
    pn1 = f(y, t)
    yn2 = y + h*pn1
    pn2 = f(yn2, t + h)
    return y + h *(pn1 + pn2) / 2
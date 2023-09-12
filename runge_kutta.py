def step_Runge_Kutta(y, t, h, f): 
    p1 = f(y,t)  
    p2 = f(y+h*p1/2., t+h/2.)
    p3 = f(y+h*p2/2., t+h/2.)
    p4 = f(y+h*p3, t+h)
    return y + h*(p1 +2*p2 + 2*p3 + p4)/6
